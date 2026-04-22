import os
import sys

# === 1. ИСПРАВЛЕНИЕ ДЛЯ STREAMLIT CLOUD (Концепция: Совместимость БД) ===
# Этот блок заменяет старый SQLite на сервере на новый, чтобы работала Memory
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
# 🔑 ЗОНА НАСТРОЕК И ЗНАНИЙ
# ==========================================
st.sidebar.header("🔑 Доступ")
api_key = st.sidebar.text_input("Введите Gemini API Key", type="password")

st.sidebar.markdown("---")
# Концепция 2: Knowledge (База знаний)
st.sidebar.header("📚 База знаний")
university_regulations = st.sidebar.text_area(
    "Регламент университета:",
    "Договор должен содержать: 1. Реквизиты сторон. 2. Сроки практики. 3. Обязанности по охране труда. "
    "Договор НЕ должен содержать: Штрафов для студента или передачи исключительных прав на разработки без компенсации."
)


# ==========================================
# 📊 СТРУКТУРА ДАННЫХ (Pydantic)
# ==========================================
class ContractAnalysis(BaseModel):
    decision: str = Field(description="Вердикт: Одобрено / На доработку / Отклонено")
    missing_points: list[str] = Field(description="Список отсутствующих обязательных элементов")
    risks: list[str] = Field(description="Найденные юридические или финансовые риски")
    summary: str = Field(description="Краткое пояснение для студента")


# ==========================================
# 🔧 ИНСТРУМЕНТЫ (Концепция 6: Tools)
# ==========================================
# Инструмент 1: Стандартный для чтения файлов
file_reader = FileReadTool()


# Инструмент 2: Кастомный для поиска ключевых условий
@tool("Legal Keywords Checker")
def check_legal_keywords(text: str) -> str:
    """Ищет наличие ключевых юридических терминов в тексте договора."""
    keywords = ["практика", "ответственность", "конфиденциальность", "срок"]
    found = [w for w in keywords if w.lower() in text.lower()]
    return f"Найдено совпадений: {', '.join(found)}. Всего: {len(found)} из {len(keywords)}."


# ==========================================
# 🖥️ ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ==========================================
st.title("📄 Автоматическая проверка договоров на практику")
st.write("Система анализирует договор, проверяет риски и сверяет его с регламентом университета.")

# Концепция 1: Files (Загрузка файла)
uploaded_file = st.file_uploader("Загрузите договор (.txt или .docx)", type=['txt', 'docx'])

if st.button("Запустить анализ") and uploaded_file:
    if not api_key:
        st.error("Пожалуйста, введите API ключ в боковой панели!")
    else:
        # Настройка ключа и стабильной версии модели
        os.environ["GEMINI_API_KEY"] = api_key
        MODEL_NAME = "gemini/gemini-1.5-flash-002"

        # Сохранение временного файла для агентов
        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Команда агентов изучает документ..."):

                # --- АГЕНТЫ ---
                analyst = Agent(
                    role="Юрист-аналитик",
                    goal="Извлечь структуру и условия договора из файла.",
                    backstory="Ты профессионал в разборе текстов. Твоя задача — найти в файле кто, с кем и на какой срок договаривается.",
                    tools=[file_reader],
                    llm=MODEL_NAME,
                    verbose=True
                )

                risk_officer = Agent(
                    role="Оценщик рисков",
                    goal="Найти опасные пункты и несоответствия регламенту.",
                    backstory="Ты защитник интересов университета и студента. Ты ищешь скрытые штрафы и нарушения правил.",
                    tools=[check_legal_keywords],
                    llm=MODEL_NAME,
                    verbose=True
                )

                coordinator = Agent(
                    role="Координатор практики",
                    goal="Сформировать итоговое юридическое заключение.",
                    backstory="Ты главный судья. Ты собираешь данные от аналитиков и выносишь вердикт.",
                    llm=MODEL_NAME,
                    verbose=True
                )

                # --- ЗАДАЧИ ---
                t1 = Task(
                    description=f"Прочитай файл {temp_filename} и выдели основные разделы.",
                    expected_output="Краткое описание структуры документа.",
                    agent=analyst
                )

                t2 = Task(
                    description=f"Сверь условия договора с этим регламентом: {university_regulations}.",
                    expected_output="Список рисков и того, чего не хватает в тексте.",
                    agent=risk_officer
                )

                # Концепция 5: Conditional Task (Условная логика)
                t3 = Task(
                    description="Если найдены риски, напиши пошаговый план исправлений. Если рисков нет, напиши 'Правки не требуются'.",
                    expected_output="План действий по доработке договора.",
                    agent=risk_officer
                )

                t4 = Task(
                    description="Собери всю информацию и подготовь структурированный отчет.",
                    expected_output="Финальный вердикт в формате Pydantic объекта.",
                    agent=coordinator,
                    output_pydantic=ContractAnalysis
                    # Концепция 4: HITL (Остановка для человека). 
                    # В Web-версии (Streamlit Cloud) этот параметр отключен для стабильности, 
                    # но в коде для локального запуска он ставится так: human_input=True
                )

                # --- ЗАПУСК ЭКИПАЖА ---
                # Концепция 3: Memory (Память для связи этапов)
                crew = Crew(
                    agents=[analyst, risk_officer, coordinator],
                    tasks=[t1, t2, t3, t4],
                    memory=True,
                    verbose=True
                )

                st.info("💡 Анализ начался. Результаты появятся ниже.")
                result = crew.kickoff()

                # --- ВЫВОД РЕЗУЛЬТАТОВ ---
                st.success("Проверка завершена!")
                if hasattr(result, 'pydantic') and result.pydantic:
                    res = result.pydantic
                    st.metric("Вердикт системы", res.decision)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🚨 Найденные риски")
                        for r in res.risks:
                            st.write(f"- {r}")
                    with col2:
                        st.subheader("❓ Что отсутствует")
                        for m in res.missing_points:
                            st.write(f"- {m}")

                    st.info(f"**Резюме для студента:** {res.summary}")
                else:
                    st.write(result.raw)

        except Exception as e:
            st.error(f"Произошла ошибка при анализе: {e}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)