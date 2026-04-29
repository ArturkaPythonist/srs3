# Анализатор обратной связи выпускников 

Этот проект готов к деплою на Streamlit Community Cloud.

## Развертывание в облаке (Streamlit Cloud)

1. Загрузи файлы `app.py` и `requirements.txt` в свой репозиторий на GitHub.
2. Зайди на [share.streamlit.io](https://share.streamlit.io/) и нажми **New app**.
3. Выбери свой репозиторий и укажи `app.py` как главный файл.
4. **Важно:** Перед тем как нажать "Deploy", нажми на **Advanced settings...**.
5. В поле **Secrets** вставь свой API-ключ в следующем формате:
   ```toml
   GOOGLE_API_KEY = "твой_ключ_AIzaSy..."