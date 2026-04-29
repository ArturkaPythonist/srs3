import streamlit as st
import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool
from langchain_google_genai import ChatGoogleGenerativeAI  # <-- Возвращаем LangChain

# Настройка страницы
st.set_page_config(page_title="Graduate Feedback Analyzer", layout="wide")
st.title("🎓 Анализатор обратной связи выпускников")

# Получение API ключа из Secrets или локально
api_key = "AIzaSyB1CdIDUMPedGOX_yF2auWzPDYupPgu814"
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]

# Принудительно устанавливаем ключ в переменные окружения,
# так как LangChain и некоторые утилиты CrewAI ищут его именно там
os.environ["GEMINI_API_KEY"] = api_key

# Используем железобетонный LangChain коннектор
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    max_tokens=2500,
    google_api_key=api_key
)

st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV (колонки: review, job)", type=["csv"])

if uploaded_file:
    # Сохраняем файл
    temp_file_path = "grad_data.csv"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Файл успешно загружен!")

    csv_tool = FileReadTool(file_path=temp_file_path)

    # Агенты с llm от LangChain
    analyst = Agent(
        role='Тематический аналитик',
        goal='Выявить ключевые категории проблем и достижений из отзывов выпускников',
        backstory='Вы эксперт по анализу текстов и оценке качества образования.',
        tools=[csv_tool],
        llm=llm,  # <-- Передаем объект LangChain
        verbose=True
    )

    career_specialist = Agent(
        role='Карьерный консультант',
        goal='Связать образовательные пробелы с текущими позициями выпускников',
        backstory='Вы анализируете влияние программы на карьерный трек.',
        llm=llm,
        verbose=True
    )

    prorector = Agent(
        role='Проректор по учебной работе',
        goal='Подготовить итоговый управленческий отчет с рекомендациями',
        backstory='Вы превращаете сырые данные в стратегические решения.',
        llm=llm,
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
        with st.spinner("Агенты работают (это может занять минуту)..."):

            try:
                base_crew = Crew(
                    agents=[analyst, career_specialist],
                    tasks=[task_analysis, task_career],
                    process=Process.sequential,
                    memory=False  # <-- Временно отключаем память для максимальной стабильности на облаке
                )

                intermediate_result = base_crew.kickoff()

                # Финальный отчет
                final_report_task = Task(
                    description="Подготовь финальный отчет: Сильные стороны, Критические слабости, Рекомендации.",
                    expected_output="Управленческий отчет в Markdown.",
                    agent=prorector,
                    context=[task_career]
                )

                final_crew = Crew(
                    agents=[prorector],
                    tasks=[final_report_task],
                    memory=False
                )

                final_output = final_crew.kickoff()

                st.markdown("---")
                st.subheader("📊 Итоговый управленческий отчет")
                st.markdown(str(final_output))

            except Exception as e:
                st.error(f"Произошла ошибка при выполнении: {e}")

else:
    st.info("Ожидание загрузки CSV файла для начала работы.")