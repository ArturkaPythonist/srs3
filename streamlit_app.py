import os
import sys

# === 1. ХАК ДЛЯ STREAMLIT CLOUD (Исправляет ошибку SQLite) ===
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
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Настройка страницы ---
st.set_page_config(page_title="Reviewer Pro (Option 13)", layout="wide")

# ==========================================
# 🔑 ЗОНА НАСТРОЕК (API & KNOWLEDGE)
# ==========================================
st.sidebar.header("🔑 Настройки системы")
api_key = st.sidebar.text_input("Введите Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📚 База знаний (Knowledge)")
university_regulations = st.sidebar.text_area(
    "Регламент университета по практике:",
    "Договор должен содержать: 1. Реквизиты сторон. 2. Сроки практики. 3. Обязанности по охране труда. "
    "Договор НЕ должен содержать: Штрафов для студента или передачи исключительных прав на разработки без компенсации."
)


# ==========================================
# 📊 PYDANTIC СХЕМА (Структурированный вывод)
# ==========================================
class ContractAnalysis(BaseModel):
    decision: str = Field(description="Вердикт: Одобрено / На доработку / Отклонено")
    missing_points: list[str] = Field(description="Список отсутствующих обязательных элементов")
    risks: list[str] = Field(description="Найденные юридические или финансовые риски")
    summary: str = Field(description="Краткое пояснение для студента")


# ==========================================
# 🔧 ИНСТРУМЕНТЫ (Tools)
# ==========================================
file_reader = FileReadTool()


@tool("Compliance Checker")
def check_legal_match(text: str) -> str:
    """Проверяет текст на соответствие ключевым юридическим терминам практики."""
    keywords = ["практика", "обучение", "договор", "стороны", "обязательства"]
    found = [word for word in keywords if word.lower() in text.lower()]
    return f"Проверка завершена. Найдено совпадений: {len(found)} из {len(keywords)}."


# ==========================================
# 🖥️ ИНТЕРФЕЙС И ЛОГИКА
# ==========================================
st.title("📄 Анализатор договоров на практику (Вариант 13)")

uploaded_file = st.file_uploader("Загрузите файл договора (.txt или .docx)", type=['txt', 'docx'])

if st.button("Запустить проверку") and uploaded_file:
    if not api_key:
        st.error("Пожалуйста, введите API ключ в боковой панели!")
    else:
        # Прямое подключение через Langchain (решает ошибку 404)
        os.environ["GOOGLE_API_KEY"] = api_key
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        # Сохраняем файл для инструмента FileReadTool (Концепция 1: Files)
        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Мультиагентная система анализирует документ..."):

                # 🤖 АГЕНТЫ
                analyst = Agent(
                    role="Юрист-аналитик",
                    goal="Извлечь структуру и условия договора из файла.",
                    backstory="Ты эксперт по документам. Твоя задача — прочитать файл и выписать условия.",
                    tools=[file_reader],
                    llm=gemini_llm,
                    verbose=True
                )

                risk_officer = Agent(
                    role="Оценщик рисков",
                    goal="Найти противоречия и опасные пункты.",
                    backstory="Ты защищаешь интересы студентов. Ты ищешь штрафы и скрытые угрозы.",
                    tools=[check_legal_match],
                    llm=gemini_llm,
                    verbose=True
                )

                coordinator = Agent(
                    role="Координатор практики",
                    goal="Сформировать финальное решение.",
                    backstory="Ты финальный судья. Ты решаешь, допускать ли договор к подписанию.",
                    llm=gemini_llm,
                    verbose=True
                )

                # 📋 ЗАДАЧИ
                t1 = Task(
                    description=f"Прочитай файл {temp_filename} и опиши ключевые разделы.",
                    expected_output="Структурированное содержание договора.",
                    agent=analyst
                )

                t2 = Task(
                    description=f"Проверь условия на соответствие регламенту: {university_regulations}.",
                    expected_output="Список рисков и отсутствующих пунктов.",
                    agent=risk_officer
                )

                # Концепция 4: Conditional Task (логика запуска)
                t3 = Task(
                    description="Если в t2 найдены риски, составь план правок. Если рисков нет, напиши 'Правки не требуются'.",
                    expected_output="Список необходимых изменений в договоре.",
                    agent=risk_officer
                )

                t4 = Task(
                    description="Подведи итог всей проверки и сформируй вердикт.",
                    expected_output="Объект со структурой Pydantic.",
                    agent=coordinator,
                    output_pydantic=ContractAnalysis,
                    human_input=True  # Концепция 5: HITL (Остановка для подтверждения человеком)
                )

                # 🚀 ЭКИПАЖ
                crew = Crew(
                    agents=[analyst, risk_officer, coordinator],
                    tasks=[t1, t2, t3, t4],
                    memory=True,  # Концепция 3: Memory
                    verbose=True
                )

                st.info(
                    "💡 Процесс запущен. Посмотри в терминал PyCharm — система ждет твоего подтверждения (HITL) перед финалом.")
                result = crew.kickoff()

                # Вывод результата
                st.success("Анализ завершен!")
                if hasattr(result, 'pydantic') and result.pydantic:
                    res = result.pydantic
                    st.metric("Вердикт", res.decision)
                    st.write("### 🚨 Найденные риски")
                    st.write(res.risks)
                    st.write("### ❓ Что отсутствует")
                    st.write(res.missing_points)
                    st.info(f"**Сводка:** {res.summary}")
                else:
                    st.write(result.raw)

        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)