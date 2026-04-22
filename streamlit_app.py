import streamlit as st
import os
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, tool

# ==========================================
# ⚙️ НАСТРОЙКА LLM (Google GenAI)
# ==========================================
# CrewAI автоматически подхватит ключ из os.environ["GEMINI_API_KEY"]
gemini_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)


# ==========================================
# 📊 PYDANTIC СХЕМЫ (Структурированный вывод)
# ==========================================
class ContractEvaluation(BaseModel):
    decision: str = Field(
        description="Итоговое решение: 'Договор можно принять', 'Нужно исправить', 'Требуется ручная проверка'")
    missing_elements: list[str] = Field(description="Список отсутствующих обязательных пунктов")
    risk_factors: list[str] = Field(description="Юридически сомнительные или рискованные места")
    final_summary: str = Field(description="Краткое резюме для сотрудника университета")


# ==========================================
# 🔧 ИНСТРУМЕНТЫ (Tools)
# ==========================================
# Стандартный тул из CrewAI (Концепция 6: Tools)
file_reader = FileReadTool()


@tool("Mandatory Clauses Checker")
def check_mandatory_clauses(text: str) -> str:
    """Кастомный инструмент для сверки текста договора с внутренним регламентом университета."""
    required = ["Ответственность сторон", "Сроки практики", "Обязанности университета"]
    missing = [req for req in required if req.lower() not in text.lower()]
    if missing:
        return f"Отсутствуют обязательные пункты: {', '.join(missing)}"
    return "Все базовые обязательные пункты присутствуют."


# ==========================================
# 🖥️ ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.set_page_config(page_title="Анализатор договоров", layout="wide")
st.title("📄 Автоматическая проверка договоров на практику")

# --- Концепция 2: Knowledge ---
st.sidebar.header("База знаний (Knowledge)")
university_regulations = st.sidebar.text_area(
    "Внутренний регламент университета:",
    "Договор обязательно должен включать сроки проведения практики, ответственность принимающей стороны и не должен содержать штрафов для студента."
)

# --- Концепция 1: Files ---
uploaded_file = st.file_uploader("Загрузите договор студента (TXT или DOCX)", type=['txt', 'docx'])

if st.button("Запустить проверку") and uploaded_file:
    # Сохраняем файл локально для FileReadTool
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Агенты изучают договор..."):

        # ==========================================
        # 🤖 АГЕНТЫ
        # ==========================================
        extractor_agent = Agent(
            role="Специалист по разбору документов",
            goal="Извлечь структуру договора и ключевые условия.",
            backstory="Ты юрист-аналитик. Твоя задача — прочитать сырой файл договора и вытащить из него суть: кто, с кем, на каких условиях.",
            tools=[file_reader],
            llm=gemini_llm,
            verbose=True
        )

        compliance_agent = Agent(
            role="Оценщик рисков и комплаенса",
            goal="Проверить наличие обязательных пунктов и найти рискованные формулировки.",
            backstory="Ты строгий проверяющий. Опираешься на внутренний регламент университета.",
            tools=[check_mandatory_clauses],
            llm=gemini_llm,
            verbose=True
        )

        clarification_agent = Agent(
            role="Агент дополнительных уточнений",
            goal="Сформировать список доработок по сомнительным местам.",
            backstory="Ты педантичный делопроизводитель. Включаешься только если договор составлен криво.",
            llm=gemini_llm,
            verbose=True
        )

        final_reviewer = Agent(
            role="Главный координатор по практике",
            goal="Подготовить финальное заключение по договору.",
            backstory="Ты принимаешь окончательное решение, допускать студента к практике по этому договору или нет.",
            llm=gemini_llm,
            verbose=True
        )

        # ==========================================
        # 📋 ЗАДАЧИ
        # ==========================================
        task_extract = Task(
            description=f"Прочитай файл {file_path}. Извлеки структуру и ключевые условия.",
            expected_output="Структурированное саммари договора.",
            agent=extractor_agent
        )

        task_check = Task(
            description=f"Опираясь на регламент: '{university_regulations}', проверь результаты предыдущего агента. Найди риски и недостающие пункты.",
            expected_output="Отчет о найденных рисках и отсутствии пунктов.",
            agent=compliance_agent
        )


        # Концепция 4: Conditional Task
        def needs_clarification(context):
            # Если compliance_agent найдет риски или нехватку пунктов, вернется True
            return True  # Для демонстрации (срабатывания условия) оставляем True. В реальности тут парсинг context.


        task_clarify = Task(
            description="Сформируй четкий список того, что нужно исправить или запросить у компании.",
            expected_output="Список пунктов на доработку.",
            agent=clarification_agent,
            condition=needs_clarification
        )

        task_final = Task(
            description="Собери данные от всех агентов и сформируй финальное заключение.",
            expected_output="Финальный статус документа в формате JSON согласно Pydantic схеме.",
            agent=final_reviewer,
            output_pydantic=ContractEvaluation,  # Интеграция Pydantic
            human_input=True  # Концепция 5: HITL (Остановка терминала для подтверждения человеком)
        )

        # ==========================================
        # 🚀 СБОРКА И ЗАПУСК
        # ==========================================
        crew = Crew(
            agents=[extractor_agent, compliance_agent, clarification_agent, final_reviewer],
            tasks=[task_extract, task_check, task_clarify, task_final],
            memory=True,  # Концепция 3: Memory
            verbose=True
        )

        st.info("Внимание! Посмотри в терминал. Перед выдачей финала сработает HITL (ожидание твоего подтверждения).")

        # Запускаем экипаж
        result = crew.kickoff()

        # Вывод структурированного Pydantic результата
        st.success("Проверка завершена!")
        if hasattr(result, 'pydantic') and result.pydantic:
            st.write("### 📝 Итоговое заключение")
            st.metric(label="Вердикт", value=result.pydantic.decision)

            st.write("**Отсутствующие пункты:**")
            for item in result.pydantic.missing_elements:
                st.write(f"- {item}")

            st.write("**Юридические риски:**")
            for risk in result.pydantic.risk_factors:
                st.write(f"- {risk}")

            st.info(f"**Резюме:** {result.pydantic.final_summary}")
        else:
            st.write(result.raw)

    # Уборка временного файла
    if os.path.exists(file_path):
        os.remove(file_path)