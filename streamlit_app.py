import streamlit as st
import os
import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM  # <-- Добавили LLM сюда
from crewai_tools import FileReadTool

# Настройка страницы
st.set_page_config(page_title="Graduate Feedback Analyzer", layout="wide")
st.title("🎓 Анализатор обратной связи выпускников")

# Метод получения API ключа
api_key = "AIzaSyB1CdIDUMPedGOX_yF2auWzPDYupPgu814"
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]

# ВАЖНО: Новая инициализация LLM для CrewAI 1.0+
# Используем внутренний класс библиотеки
gemini_llm = LLM(
    model="gemini-1.5-pro",
    api_key=api_key,
    temperature=0.3
)

# 1. Загрузка данных
st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV (колонки: review, job)", type=["csv"])

if uploaded_file:
    # Сохранение временного файла для инструментов CrewAI
    temp_file_path = "grad_data.csv"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Файл успешно загружен!")

    # Инструменты
    csv_tool = FileReadTool(file_path=temp_file_path)

    # Определение Агентов (передаем правильный gemini_llm)
    analyst = Agent(
        role='Тематический аналитик',
        goal='Выявить ключевые категории проблем и достижений из отзывов выпускников',
        backstory='Вы эксперт по анализу текстов и оценке качества образования. Ваша задача — категоризировать отзывы.',
        tools=[csv_tool],
        llm=gemini_llm,
        verbose=True
    )

    career_specialist = Agent(
        role='Карьерный консультант',
        goal='Связать образовательные пробелы с текущими позициями выпускников',
        backstory='Вы анализируете, как отсутствие определенных навыков в программе влияет на карьерный трек.',
        llm=gemini_llm,
        verbose=True
    )

    prorector = Agent(
        role='Проректор по учебной работе',
        goal='Подготовить итоговый управленческий отчет с рекомендациями',
        backstory='Вы превращаете сырые данные в стратегические решения для университета.',
        llm=gemini_llm,
        verbose=True
    )

    # Определение Задач
    task_analysis = Task(
        description="Проанализируй отзывы из файла. Категоризируй их: Инфраструктура, Актуальность, Преподавание.",
        expected_output="Структурированный список тем с примерами отзывов.",
        agent=analyst
    )

    task_career = Task(
        description="Сопоставь найденные проблемы с текущими должностями выпускников (колонки job).",
        expected_output="Отчет о влиянии программы на карьеру.",
        agent=career_specialist,
        context=[task_analysis]
    )

    if st.button("Запустить анализ агентами"):
        with st.spinner("Агенты обрабатывают данные..."):

            # Первый этап: Базовый анализ
            base_crew = Crew(
                agents=[analyst, career_specialist],
                tasks=[task_analysis, task_career],
                process=Process.sequential,
                memory=True
            )

            intermediate_result = base_crew.kickoff()

            # Логика проверки результата
            result_str = str(intermediate_result).lower()
            needs_more = any(word in result_str for word in ["противореч", "недостаточно", "неясно", "uncertain"])

            if needs_more:
                st.warning("Обнаружены противоречивые данные. Запуск уточняющего анализа...")
                refinement_task = Task(
                    description="Найдены противоречия. Проведи повторный глубокий анализ по спорным категориям.",
                    expected_output="Уточненная сводка данных.",
                    agent=analyst
                )
                refine_crew = Crew(agents=[analyst], tasks=[refinement_task])
                intermediate_result = refine_crew.kickoff()

            # Финальный отчет
            final_report_task = Task(
                description="Подготовь финальный отчет: Сильные стороны, Критические слабости, Рекомендации.",
                expected_output="Управленческий отчет в формате Markdown.",
                agent=prorector,
                context=[task_career]
            )

            final_crew = Crew(
                agents=[prorector],
                tasks=[final_report_task]
            )

            final_output = final_crew.kickoff()

            st.markdown("---")
            st.subheader("📊 Итоговый управленческий отчет")

            # Приводим вывод к строке для безопасности новой версии
            st.markdown(str(final_output))

else:
    st.info("Ожидание загрузки CSV файла для начала работы.")