# Telegram-бот к языковой модели

Принимает текстовое сообщение, отправляет его в языковую модель, возвращает ответ в чат.

**Без памяти.** Каждое сообщение — независимый запрос `system prompt + текст пользователя`.
История диалога не хранится и модели не передаётся.

**Два бэкенда**, переключаются переменной `LLM_PROVIDER`, код не меняется:

| Значение | Модель | Требует |
|---|---|---|
| `ollama` *(по умолчанию)* | локальная `qwen3:1.7b` | запущенную Ollama |
| `anthropic` | облачный `claude-opus-5` | `ANTHROPIC_API_KEY` |

---

## Требования

Python 3.11+, [uv](https://docs.astral.sh/uv/), токен бота от [@BotFather](https://t.me/BotFather).
Для локального профиля — [Ollama](https://ollama.com).

## Установка

```bash
git clone https://github.com/andreykutsenko/tg-llm-agent.git
cd tg-llm-agent

uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Нет `uv` — поставить одной командой:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

## Настройка

```bash
cp .env.example .env
```

В `.env` заполнить **одну строку** — `TELEGRAM_BOT_TOKEN`.
Остальные переменные имеют рабочие значения по умолчанию.

Для облачного профиля дополнительно: раскомментировать блок `anthropic`
и задать `ANTHROPIC_API_KEY`.

## Запуск

Локальный профиль требует двух терминалов.

**Терминал 1 — сервер модели.** Держать открытым:

```bash
ollama serve
```

**Терминал 2 — скачать модель (один раз) и запустить бота:**

```bash
ollama pull qwen3:1.7b
.venv/bin/python bot.py
```

Для облачного профиля Ollama не нужна — достаточно второй команды.

Транспорт — long polling, публичный IP и открытый порт не требуются.
Первый запрос после старта Ollama идёт дольше: модель грузится в память,
поэтому таймаут 180 секунд.

## Проверка

```bash
.venv/bin/python -m pytest      # 22 теста, сеть не используется
```

В Telegram: `/start` покажет активную модель, любое текстовое сообщение вернёт ответ.
Смена `LLM_PROVIDER` в `.env` и перезапуск переключают бота на другую модель —
код при этом не меняется.

Если модель недоступна, бот не падает: отвечает подсказкой, что проверить.

## Структура

```
bot.py             хендлеры, нарезка ответа под лимит 4096, polling
config.py          чтение и валидация переменных окружения
llms/__init__.py   call_llm(prompt) -> str: выбор провайдера, вырезание <think>
llms/ollama.py     локальная модель, AsyncOpenAI
llms/anthropic.py  облачная модель, AsyncAnthropic
llms/errors.py     LLMError / LLMTimeoutError
tests/             pytest, оба клиента замоканы
```

`bot.py` знает только про `call_llm`. Третий провайдер — новый файл в `llms/`
и одна запись в `_GENERATORS`.

---

Спецификация, по которой сгенерирован проект — [SPEC-tg-bot.md](SPEC-tg-bot.md).
Замеры, развёртывание и принятые решения — [REPORT.md](REPORT.md).
