import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool
from google import genai

# Настройка страницы
st.set_page_config(page_title="Graduate Feedback Analyzer", layout="wide")
st.title("🎓 Анализатор обратной связи выпускников")

# БЕЗОПАСНОЕ ПОЛУЧЕНИЕ КЛЮЧА
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Ошибка: API ключ не найден в Secrets. Добавьте GOOGLE_API_KEY в настройках Streamlit Cloud.")
    st.stop()

os.environ["GEMINI_API_KEY"] = api_key

# === СКАНЕР МОДЕЛЕЙ (ОТЛАДКА API) ===
st.sidebar.markdown("### 🛠 Отладка API")
if st.sidebar.button("Просканировать доступные модели"):
    with st.sidebar.status("Стучимся на сервер Google..."):
        try:
            client = genai.Client(api_key=api_key)
            # Вытягиваем только те модели, которые умеют генерировать текст
            models = [m.name.replace("models/", "") for m in client.models.list() if
                      "generateContent" in m.supported_generation_methods]
            st.success("Сканирование завершено!")
            st.write("**Разрешенные модели для твоего ключа:**")
            for m in models:
                st.code(f"gemini/{m}")
        except Exception as e:
            st.error(f"Ошибка сканирования: {e}")
# =====================================

st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV (колонки: review, job)", type=["csv"])

if uploaded_file:
    # Сохраняем файл для инструментов
    temp_file_path = "grad_data.csv"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Файл успешно загружен!")

    csv_tool = FileReadTool(file_path=temp_file_path)

    # Агенты: используем точную версию с припиской -002 (наиболее стабильная)
    analyst = Agent(
        role='Тематический аналитик',
        goal='Выявить ключевые категории проблем и достижений из отзывов',
        backstory='Вы эксперт по анализу текстов и оценке качества образования.',
        tools=[csv_tool],
        llm="gemini/gemini-1.5-flash-002",
        verbose=True
    )

    career_specialist = Agent(
        role='Карьерный консультант',
        goal='Связать образовательные пробелы с текущими позициями выпускников',
        backstory='Вы анализируете влияние программы на карьерный трек.',
        llm="gemini/gemini-1.5-flash-002",
        verbose=True
    )

    prorector = Agent(
        role='Проректор по учебной работе',
        goal='Подготовить итоговый управленческий отчет с рекомендациями',
        backstory='Вы превращаете сырые данные в стратегические решения.',
        llm="gemini/gemini-1.5-flash-002",
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
        with st.spinner("Агенты анализируют данные..."):
            try:
                base_crew = Crew(
                    agents=[analyst, career_specialist],
                    tasks=[task_analysis, task_career],
                    process=Process.sequential
                )

                intermediate_result = base_crew.kickoff()

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
                st.markdown(str(final_output))

            except Exception as e:
                st.error(f"Произошла ошибка при выполнении: {e}")

else:
    st.info("Ожидание загрузки CSV файла.")