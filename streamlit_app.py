import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Устанавливаем ключ напрямую для быстрого старта (ВНИМАНИЕ: не для продакшена!)
os.environ["GOOGLE_API_KEY"] = "AIzaSyB1CdIDUMPedGOX_yF2auWzPDYupPgu814"

# Настройка стабильной модели Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
    max_tokens=1024
)

st.set_page_config(page_title="CrewAI + Streamlit", page_icon="🤖")
st.title("🤖 Моя первая команда CrewAI")
st.markdown("Это минимальное рабочее приложение на Streamlit.")

# Поле для ввода темы
topic = st.text_input("Введите тему для небольшого исследования:",
                      "Влияние искусственного интеллекта на кибербезопасность")

if st.button("Запустить агентов"):
    with st.spinner("Агенты работают..."):
        try:
            # 1. Создаем агентов
            researcher = Agent(
                role='Старший исследователь',
                goal=f'Найти ключевые факты по теме: {topic}',
                backstory='Вы — опытный аналитик, умеющий находить суть в любой теме.',
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            writer = Agent(
                role='Технический писатель',
                goal=f'Написать краткое и понятное резюме на основе исследования по теме: {topic}',
                backstory='Вы умеете объяснять сложные концепции простым языком.',
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            # 2. Создаем задачи
            task1 = Task(
                description=f'Собери 3 главных факта о: {topic}.',
                expected_output='Список из 3 ключевых фактов.',
                agent=researcher
            )

            task2 = Task(
                description='Напиши один абзац (до 500 символов), обобщающий факты.',
                expected_output='Связный абзац текста.',
                agent=writer
            )

            # 3. Собираем команду (Crew)
            crew = Crew(
                agents=[researcher, writer],
                tasks=[task1, task2],
                process=Process.sequential
            )

            # 4. Запускаем выполнение
            result = crew.kickoff()

            st.success("Готово!")
            st.subheader("Результат:")
            st.write(result)

        except Exception as e:
            st.error(f"Произошла ошибка: {e}")