.PHONY: help install dev test lint format run docker-up docker-down clean

help:
	@echo "Команды для разработки fastapi-ai-enterprise-starter:"
	@echo "  make install    - Установить зависимости в виртуальное окружение"
	@echo "  make dev        - Запустить сервер разработки с hot-reload"
	@echo "  make test       - Запустить набор pytest тестов"
	@echo "  make lint       - Проверить код линтером ruff"
	@echo "  make format     - Отформатировать код ruff format"
	@echo "  make docker-up  - Поднять окружение (сервис + Redis) в Docker"
	@echo "  make docker-down- Остановить Docker контейнеры"
	@echo "  make clean      - Очистить кэш Python и pytest"

install:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .

format:
	ruff format .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
