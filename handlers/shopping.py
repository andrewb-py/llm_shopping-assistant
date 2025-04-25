from aiogram import Router, types
from aiogram.filters import Command
from shopping_assistant import ShoppingAssistant
from config import Config

router = Router()
assistant = ShoppingAssistant(Config.CSV_PATH, Config.OPENROUTER_API_KEY)

@router.message(Command("reset"))
async def reset_session(message: types.Message):
    assistant.reset_session()
    await message.answer("Сессия сброшена. Начинаем новый диалог.")

@router.message()
async def handle_message(message: types.Message):
    user_message = message.text
    response = assistant.process_user_message(user_message)
    await message.answer(response)