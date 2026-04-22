import os
import sys

# === 1. ИСПРАВЛЕНИЕ ДЛЯ STREAMLIT CLOUD (SQLite) ===
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew
from crewai.tools import tool
from crewai_tools import FileReadTool

# --- Настройка интерфейса ---
st.set_page_config(page_title="Анализатор договоров (Вариант 13)", layout="wide")

# ==========================================
# 🔑 ЗОНА НАСТРОЕК
# ==========================================
st.sidebar.header("🔑 Доступ")
api_key = st.sidebar.text_input("Введите свежий Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📚 База знаний")
university_regulations = st.sidebar.text_area(
    "Регламент университета:",
    "Договор должен содержать: 1. Реквизиты сторон. 2. Сроки практики. 3. Обязанности по охране труда. "
    "Договор НЕ должен содержать: Штрафов для студента или передачи прав без компенсации."
)


# --- Структура вывода (Pydantic) ---
class ContractAnalysis(BaseModel):
    decision: str = Field(description="Вердикт: Одобрено / На доработку / Отклонено")
    missing_points: list[str] = Field(description="Список отсутствующих элементов")
    risks: list[str] = Field(description="Найденные риски")
    summary: str = Field(description="Резюме для студента")


# --- Инструменты ---
file_reader = FileReadTool()


@tool("Legal Keywords Checker")
def check_legal_keywords(text: str) -> str:
    """Ищет наличие ключевых юридических терминов."""
    keywords = ["практика", "ответственность", "срок", "обучение"]
    found = [w for w in keywords if w.lower() in text.lower()]
    return f"Найдено совпадений: {len(found)} из {len(keywords)}."


# ==========================================
# 🖥️ ЛОГИКА ПРИЛОЖЕНИЯ
# ==========================================
st.title("📄 Анализатор договоров на практику")

uploaded_file = st.file_uploader("Загрузите договор (.txt или .docx)", type=['txt', 'docx'])

if st.button("Запустить анализ") and uploaded_file:
    if not api_key:
        st.error("Ошибка: Введите API ключ в боковой панели!")
    else:
        # Устанавливаем ключ
        os.environ["GEMINI_API_KEY"] = api_key
        # Используем стандартную модель для свежего ключа
        MODEL_NAME = "gemini/gemini-1.5-pro"

        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Агенты работают..."):

                # --- АГЕНТЫ ---
                analyst = Agent(
                    role="Юрист-аналитик",
                    goal="Извлечь условия договора.",
                    backstory="Ты профи по документам.",
                    tools=[file_reader],
                    llm=MODEL_NAME,
                    verbose=True
                )

                risk_officer = Agent(
                    role="Оценщик рисков",
                    goal="Найти нарушения регламента.",
                    backstory="Ты защищаешь студента.",
                    tools=[check_legal_keywords],
                    llm=MODEL_NAME,
                    verbose=True
                )

                coordinator = Agent(
                    role="Координатор",
                    goal="Вынести вердикт.",
                    backstory="Ты принимаешь решение.",
                    llm=MODEL_NAME,
                    verbose=True
                )

                # --- ЗАДАЧИ ---
                t1 = Task(description=f"Разбери файл {temp_filename}.", expected_output="Структура договора.",
                          agent=analyst)

                t2 = Task(description=f"Сверь с регламентом: {university_regulations}.",
                          expected_output="Список проблем.", agent=risk_officer)

                t3 = Task(description="Если есть риски, напиши план правок. Если нет - 'Все ок'.",
                          expected_output="План действий.", agent=risk_officer)

                t4 = Task(description="Выдай финальный вердикт.", expected_output="Pydantic объект.", agent=coordinator,
                          output_pydantic=ContractAnalysis)

                # --- ЗАПУСК ---
                crew = Crew(agents=[analyst, risk_officer, coordinator], tasks=[t1, t2, t3, t4], memory=True,
                            verbose=True)

                result = crew.kickoff()

                # --- ВЫВОД ---
                st.success("Готово!")
                if hasattr(result, 'pydantic') and result.pydantic:
                    res = result.pydantic
                    st.metric("Вердикт", res.decision)
                    st.write("**Риски:**", res.risks)
                    st.write("**Отсутствует:**", res.missing_points)
                    st.info(f"**Резюме:** {res.summary}")
                else:
                    st.write(result.raw)

        except Exception as e:
            st.error(f"Ошибка анализа: {e}")
            st.info("Если видите ошибку 400 - проверьте, что ключ точно активен в Google AI Studio.")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)