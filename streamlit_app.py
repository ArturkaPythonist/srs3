import streamlit as st
import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool

# Настройка страницы
st.set_page_config(page_title="Graduate Feedback Analyzer", layout="wide")
st.title("🎓 Анализатор обратной связи выпускников")

# Получение API ключа (твой новый ключ из Secrets)
api_key = "AIzaSyB1CdIDUMPedGOX_yF2auWzPDYupPgu814"
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]

# КРИТИЧЕСКИ ВАЖНО: CrewAI ищет переменную GEMINI_API_KEY для строки "gemini/..."
os.environ["GEMINI_API_KEY"] = api_key

st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV (колонки: review, job)", type=["csv"])

if uploaded_file:
    # Сохраняем файл
    temp_file_path = "grad_data.csv"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Файл успешно загружен!")

    csv_tool = FileReadTool(file_path=temp_file_path)

    # Агенты: передаем модель просто КАК СТРОКУ
    analyst = Agent(
        role='Тематический аналитик',
        goal='Выявить ключевые категории проблем и достижений из отзывов',
        backstory='Вы эксперт по анализу текстов и оценке качества образования.',
        tools=[csv_tool],
        llm="gemini/gemini-1.5-pro",  # <-- Вот оно, решение проблемы Pydantic!
        verbose=True
    )

    career_specialist = Agent(
        role='Карьерный консультант',
        goal='Связать образовательные пробелы с текущими позициями выпускников',
        backstory='Вы анализируете влияние программы на карьерный трек.',
        llm="gemini/gemini-1.5-pro",  # <-- Модель строкой
        verbose=True
    )

    prorector = Agent(
        role='Проректор по учебной работе',
        goal='Подготовить итоговый управленческий отчет с рекомендациями',
        backstory='Вы превращаете сырые данные в стратегические решения.',
        llm="gemini/gemini-1.5-pro",  # <-- Модель строкой
        verbose=True
    )

    task_analysis = Task(
        description="Проанализируй отзывы из файла grad_data.csv. Категоризируй их (Инфраструктура, Актуальность, Преподавание).",
        expected_output="Структурированный список тем с примерами.",
        agent=analyst
    )

    task_career = Task(
        description="Сопоставь найденные проблемы с текущими должностями выпускников (колонки job).",
        expected_output="Отчет о влиянии программы на карьеру.",
        agent=career_specialist,
        context=[task_analysis]
    )

    if st.button("Запустить анализ агентами"):
        with st.spinner("Агенты работают (это может занять около минуты)..."):