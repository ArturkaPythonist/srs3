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
from crewai import Agent, Task, Crew
from crewai.tools import tool
from crewai_tools import FileReadTool
# ИМПОРТИРУЕМ ПРЯМОЙ КОННЕКТОР
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Настройка страницы ---
st.set_page_config(page_title="Reviewer Pro 2026", layout="wide")

# ==========================================
# 🔑 СЕКРЕТЫ И НАСТРОЙКИ
# ==========================================
# Пытаемся взять из Secrets, если нет — из ввода
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
    missing_points: list[str] = Field(description="Что отсутствует")
    risks: list[str] = Field(description="Риски")
    summary: str = Field(description="Итог")


# --- Инструменты ---
file_reader = FileReadTool()


@tool("Legal Checker")
def legal_tool(text: str) -> str:
    """Проверка ключевых слов."""
    keywords = ["практика", "срок", "договор"]
    found = [w for w in keywords if w.lower() in text.lower()]
    return f"Найдено: {len(found)}"


# ==========================================
# 🖥️ ОСНОВНОЙ БЛОК
# ==========================================
st.title("📄 Анализатор договоров (Вариант 13)")
uploaded_file = st.file_uploader("Загрузите файл", type=['txt', 'docx'])

if st.button("Начать анализ") and uploaded_file:
    if not api_key:
        st.error("Нет ключа!")
    else:
        # ПРЯМАЯ НАСТРОЙКА МОДЕЛИ (ОБХОД ОШИБКИ 404)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2
        )

        temp_name = f"temp_{uploaded_file.name}"
        with open(temp_name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Агенты обсуждают договор..."):

                # АГЕНТЫ (теперь используют объект llm напрямую)
                analyst = Agent(
                    role="Юрист",
                    goal="Найти условия в файле.",
                    backstory="Ты профи по текстам.",
                    tools=[file_reader],
                    llm=llm,
                    verbose=True
                )

                risk_officer = Agent(
                    role="Риск-менеджер",
                    goal="Найти нарушения.",
                    backstory="Ты защищаешь студента.",
                    tools=[legal_tool],
                    llm=llm,
                    verbose=True
                )

                # ЗАДАЧИ
                t1 = Task(description=f"Прочитай {temp_name}", expected_output="Условия.", agent=analyst)
                t2 = Task(description=f"Сверь с {university_regulations}", expected_output="Риски.", agent=risk_officer)
                t3 = Task(description="Выдай вердикт.", expected_output="Pydantic объект.", agent=risk_officer,
                          output_pydantic=ContractAnalysis)

                crew = Crew(agents=[analyst, risk_officer], tasks=[t1, t2, t3], memory=True)
                result = crew.kickoff()

                st.success("Готово!")
                if hasattr(result, 'pydantic') and result.pydantic:
                    res = result.pydantic
                    st.metric("Решение", res.decision)
                    st.write("**Риски:**", res.risks)
                    st.info(f"**Суть:** {res.summary}")
                else:
                    st.write(result.raw)

        except Exception as e:
            st.error(f"Ошибка: {e}")
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)