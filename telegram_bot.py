# telegram_bot.py — Telegram interface for the Study Assistant Agent.

import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_core.messages import HumanMessage

# Import the compiled graph from main.py
from main import study_agent

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Initialize study history per user (for now, a single shared one)
study_history = {
    "topics_covered": [],
    "correct_answers": 0,
    "wrong_answers": 0,
    "weak_topics": []
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /start command."""
    await update.message.reply_text(
        "Hello! I'm your Study Assistant 📚\n\n"
        "Ask me a question about your study material and I'll answer!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for regular text messages — passes them to the study agent."""
    user_input = update.message.text
    
    # Let the user know we're working on it
    await update.message.reply_text("🤔 I'm thinking...")
    
    # Invoke the study agent
    messages = [HumanMessage(content=user_input)]
    result = study_agent.invoke({
        "messages": messages,
        "study_history": study_history,
        "current_mode": "chat"
    })
    
    # Update study history
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                topic = tc['args'].get('query') or tc['args'].get('topic')
                if topic and topic not in study_history["topics_covered"]:
                    study_history["topics_covered"].append(topic)
    
    # Send the response back to the user
    response = result['messages'][-1].content
    await update.message.reply_text(response)


def main():
    """Start the Telegram bot."""
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot Telegram started! Go to Telegram and message your bot.")
    app.run_polling()


if __name__ == "__main__":
    main()