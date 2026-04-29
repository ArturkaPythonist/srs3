import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Анализатор выпускников", layout="wide")
st.title("🎓 Аналитическая система «Обратная связь выпускников»")

# Безопасное получение ключа для облака
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("Ключ API не найден. Настройте Secrets в панели Streamlit!")
    st.stop()

# Инициализация стабильного Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    max_tokens=2000,
    google_api_key=api_key
)

# 1. Загрузка данных
uploaded_file = st.file_uploader("Загрузите CSV с отзывами (колонки: review, job)", type=["csv"])

if uploaded_file:
    # Сохраняем файл временно для текущей сессии
    with open("temp_feedback.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Файл загружен. Начинаем анализ...")

    csv_tool = FileReadTool(file_path='temp_feedback.csv')

    analyst = Agent(
        role='Тематический аналитик',
        goal='Выделить ключевые темы и категории проблем из отзывов',
        backstory='Вы эксперт по качественному анализу текста. Сопоставляете отзывы с моделью качества образования.',
        tools=[csv_tool],
        llm=llm,
        verbose=True,
        memory=True
    )

    career_specialist = Agent(
        role='Карьерный стратег',
        goal='Связать проблемы обучения с карьерными успехами выпускников',
        backstory='Вы анализируете, как пробелы в знаниях влияют на текущие позиции выпускников.',
        llm=llm,
        verbose=True,
        memory=True
    )

    report_expert = Agent(
        role='Проректор по развитию',
        goal='Подготовить итоговый управленческий отчет',
        backstory='Вы принимаете решения. Вам нужен четкий список сильных сторон, слабостей и рекомендаций.',
        llm=llm,
        verbose=True
    )

    task1 = Task(
        description="Проанализируй temp_feedback.csv. Выдели основные жалобы и похвалу по категориям: Инфраструктура, Актуальность, Преподавание.",
        expected_output="Список тематических кластеров с цитатами.",
        agent=analyst
    )

    task2 = Task(
        description="Соотнеси отзывы из задачи 1 с текущими должностями выпускников (job). Найди закономерности.",
        expected_output="Аналитическая записка о связи программы с карьерой.",
        agent=career_specialist,
        context=[task1]
    )

    if st.button("Запустить полный цикл анализа"):
        with st.spinner("Агенты изучают данные..."):

            analysis_crew = Crew(
                agents=[analyst, career_specialist],
                tasks=[task1, task2],
                process=Process.sequential,
                memory=True
            )

            intermediate_result = analysis_crew.kickoff()

            # Условная задача (Conditional Task)
            needs_refinement = "недостаточно" in str(intermediate_result).lower() or "противореч" in str(
                intermediate_result).lower()

            if needs_refinement:
                st.warning("Обнаружены противоречивые сигналы. Запущена задача уточняющего анализа...")
                refinement_task = Task(
                    description="Проведи глубокое погружение в найденные противоречия. Перепроверь данные еще раз.",
                    expected_output="Уточненные данные по спорным категориям.",
                    agent=analyst
                )
                refine_crew = Crew(agents=[analyst], tasks=[refinement_task])
                refine_crew.kickoff()

            st.subheader("📝 Проект управленческого отчета подготовлен")

            # Финальный отчет (human_input=False для работы в облаке)
            final_task = Task(
                description="Сформируй финальный отчет: 1. Сильные стороны 2. Слабости 3. Рекомендации.",
                expected_output="Структурированный текст отчета.",
                agent=report_expert,
                human_input=False
            )

            report_crew = Crew(
                agents=[report_expert],
                tasks=[final_task],
                context=[task2]
            )

            final_report = report_crew.kickoff()

            st.markdown("### Итоговый результат:")
            st.write(final_report)

else:
    st.info("Пожалуйста, загрузите CSV файл. Пример формата: review, job")