import os
import sys

# === 1. ХАК ДЛЯ SQLITE (Streamlit Cloud) ===
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM  # Импортируем LLM напрямую из crewai
from crewai.tools import tool
from crewai_tools import FileReadTool

# --- Настройка страницы ---
st.set_page_config(page_title="Reviewer Pro 2026", layout="wide")

# ==========================================
# 🔑 СЕКРЕТЫ И НАСТРОЙКИ
# ==========================================
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Введите Gemini API Key", type="password")

st.sidebar.markdown("---")
university_regulations = st.sidebar.text_area(
    "Регламент университета:",
    "Договор должен содержать: реквизиты, сроки, охрану труда. "
    "Запрещено: штрафы для студента, отчуждение прав без оплаты."
)


# --- Структура Pydantic ---
class ContractAnalysis(BaseModel):
    decision: str = Field(description="Вердикт: Одобрено / На доработку / Отклонено")
    missing_points: list[str] = Field(description="Что отсутствует в договоре")
    risks: list[str] = Field(description="Юридические риски")
    summary: str = Field(description="Итоговое резюме")


# --- Инструменты ---
file_reader = FileReadTool()


@tool("Legal Checker")
def legal_tool(text: str) -> str:
    """Проверка наличия базовых юридических терминов."""
    keywords = ["практика", "срок", "договор", "обязанности"]
    found = [w for w in keywords if w.lower() in text.lower()]
    return f"Проверка выполнена. Найдено ключевых слов: {len(found)}"


# ==========================================
# 🖥️ ОСНОВНОЙ БЛОК ПРИЛОЖЕНИЯ
# ==========================================
st.title("📄 Анализатор договоров на практику (Вариант 13)")
uploaded_file = st.file_uploader("Загрузите файл договора (.txt или .docx)", type=['txt', 'docx'])

if st.button("Начать анализ") and uploaded_file:
    if not api_key:
        st.error("Ошибка: API ключ не найден!")
    else:
        # Устанавливаем ключ в окружение
        os.environ["GEMINI_API_KEY"] = api_key

        # ФИКС ОШИБКИ 404:
        # Используем "custom_llm_provider", чтобы LiteLLM не тупил с v1beta
        my_llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta"  # Явно указываем адрес
        )

        temp_name = f"temp_{uploaded_file.name}"
        with open(temp_name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Агенты изучают документ..."):

                # --- АГЕНТЫ ---
                analyst = Agent(
                    role="Юрист-аналитик",
                    goal="Вычитать договор и найти все условия.",
                    backstory="Ты опытный юрист. Твоя задача — сухие факты из документа.",
                    tools=[file_reader],
                    llm=my_llm,
                    verbose=True
                )

                risk_officer = Agent(
                    role="Риск-менеджер",
                    goal="Найти нарушения и риски для студента.",
                    backstory="Ты защищаешь интересы студентов и университета.",
                    tools=[legal_tool],
                    llm=my_llm,
                    verbose=True
                )

                # --- ЗАДАЧИ ---
                t1 = Task(
                    description=f"Прочитай файл {temp_name} и выпиши условия практики.",
                    expected_output="Список условий договора.",
                    agent=analyst
                )

                t2 = Task(
                    description=f"Сверь условия с регламентом: {university_regulations}.",
                    expected_output="Список найденных рисков и нарушений.",
                    agent=risk_officer
                )

                t3 = Task(
                    description="Сформируй финальный отчет на основе анализа.",
                    expected_output="Pydantic объект с вердиктом.",
                    agent=risk_officer,
                    output_pydantic=ContractAnalysis
                )

                # --- ЭКИПАЖ ---
                crew = Crew(
                    agents=[analyst, risk_officer],
                    tasks=[t1, t2, t3],
                    memory=True,
                    verbose=True
                )

                result = crew.kickoff()

                # --- ВЫВОД РЕЗУЛЬТАТОВ ---
                st.success("Анализ завершен успешно!")

                # Проверка наличия структурированного вывода
                res_data = result.pydantic if hasattr(result, 'pydantic') and result.pydantic else None

                if res_data:
                    st.metric("Вердикт", res_data.decision)
                    st.subheader("🚨 Найденные риски")
                    st.write(res_data.risks)
                    st.subheader("❓ Что нужно добавить")
                    st.write(res_data.missing_points)
                    st.info(f"**Резюме:** {res_data.summary}")
                else:
                    st.write(result.raw)

        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
            st.info("Убедитесь, что ваш API ключ активен и поддерживает модель Gemini 1.5 Flash.")
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)