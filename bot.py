from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Optional
import google.generativeai as genai
import telebot
from telebot.types import Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from pymongo import MongoClient
import requests
from PIL import Image
import io
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from googlesearch import search
import pandas as pd
import logging
import ssl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration with error checking
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')

# Validate environment variables
if not TELEGRAM_TOKEN:
    raise ValueError("No TELEGRAM_TOKEN found in .env file")
if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found in .env file")
if not MONGO_URI:
    raise ValueError("No MONGO_URI found in .env file")

# Initialize services with error handling
try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN)
    logger.info(f"Bot initialized with token starting with: {TELEGRAM_TOKEN[:5]}...")
except Exception as e:
    logger.error(f"Failed to initialize Telegram bot: {e}")
    raise

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    vision_model = genai.GenerativeModel('gemini-pro-vision')
    logger.info("Gemini API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}")
    raise

try:
    mongo_client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tls=True,
        tlsAllowInvalidCertificates=True
    )
    db = mongo_client['telegram_bot']
    # Test the connection
    mongo_client.admin.command('ping')
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

class UserManager:
    @staticmethod
    def register_user(message: Message) -> None:
        """Register a new user in MongoDB."""
        try:
            users = db.users
            user_data = {
                'chat_id': message.chat.id,
                'username': message.from_user.username,
                'first_name': message.from_user.first_name,
                'joined_at': datetime.utcnow(),
                'phone_number': None,
                'total_messages': 0,
                'last_active': datetime.utcnow()
            }
            users.update_one(
                {'chat_id': message.chat.id},
                {'$setOnInsert': user_data},
                upsert=True
            )
            logger.info(f"User registered: {message.chat.id}")
        except Exception as e:
            logger.error(f"Failed to register user: {e}")
            raise

    @staticmethod
    def save_phone_number(chat_id: int, phone_number: str) -> None:
        """Save user's phone number."""
        try:
            db.users.update_one(
                {'chat_id': chat_id},
                {'$set': {'phone_number': phone_number}}
            )
            logger.info(f"Phone number saved for user: {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save phone number: {e}")
            raise

class ChatHistory:
    @staticmethod
    def save_message(chat_id: int, user_message: str, bot_response: str) -> None:
        """Save chat history to MongoDB."""
        try:
            db.chat_history.insert_one({
                'chat_id': chat_id,
                'user_message': user_message,
                'bot_response': bot_response,
                'timestamp': datetime.utcnow()
            })
            db.users.update_one(
                {'chat_id': chat_id},
                {
                    '$inc': {'total_messages': 1},
                    '$set': {'last_active': datetime.utcnow()}
                }
            )
            logger.info(f"Chat history saved for user: {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
            raise

class FileAnalyzer:
    @staticmethod
    def analyze_image(file_path: str) -> str:
        """Analyze image using Gemini Vision."""
        try:
            image = Image.open(file_path)
            response = vision_model.generate_content(image)
            return response.text
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return "Sorry, I couldn't analyze this image. Please try again."

    @staticmethod
    def analyze_pdf(file_path: str) -> str:
        """Extract text from PDF and analyze using Gemini."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            response = model.generate_content(f"Analyze this PDF content: {text[:2000]}")
            return response.text
        except Exception as e:
            logger.error(f"Failed to analyze PDF: {e}")
            return "Sorry, I couldn't analyze this PDF. Please try again."

    @staticmethod
    def save_file_metadata(chat_id: int, filename: str, description: str) -> None:
        """Save file metadata to MongoDB."""
        try:
            db.files.insert_one({
                'chat_id': chat_id,
                'filename': filename,
                'description': description,
                'timestamp': datetime.utcnow()
            })
            logger.info(f"File metadata saved for user: {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")
            raise

class WebSearcher:
    @staticmethod
    def search_and_summarize(query: str) -> tuple[str, list[str]]:
        """Perform web search and return AI summary with links."""
        try:
            links = list(search(query, num_results=5))
            summaries = []
            
            for link in links[:3]:
                try:
                    response = requests.get(link, timeout=5)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    content = soup.get_text()[:1000]
                    summaries.append(content)
                except Exception as e:
                    logger.warning(f"Failed to fetch content from {link}: {e}")
                    continue
            
            if not summaries:
                return "I couldn't find any relevant information.", links
            
            combined_content = "\n".join(summaries)
            response = model.generate_content(
                f"Summarize these search results for '{query}': {combined_content}"
            )
            
            return response.text, links
        except Exception as e:
            logger.error(f"Failed to perform web search: {e}")
            return "Sorry, I couldn't perform the web search at this time.", []

# Bot command handlers
@bot.message_handler(commands=['start'])
def start(message: Message):
    """Handle /start command."""
    try:
        UserManager.register_user(message)
        
        markup = ReplyKeyboardMarkup(one_time_keyboard=True)
        button = KeyboardButton('Share Contact', request_contact=True)
        markup.add(button)
        
        welcome_text = (
            "Welcome! 👋 I'm your AI assistant. I can help you with:\n"
            "• General questions and chat\n"
            "• Image and file analysis\n"
            "• Web searches\n\n"
            "Please share your contact to get started!"
        )
        bot.reply_to(message, welcome_text, reply_markup=markup)
        logger.info(f"Start command processed for user: {message.chat.id}")
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        bot.reply_to(message, "Sorry, something went wrong. Please try again later.")

@bot.message_handler(content_types=['contact'])
def handle_contact(message: Message):
    """Handle shared contact information."""
    try:
        if message.contact is not None:
            UserManager.save_phone_number(message.chat.id, message.contact.phone_number)
            bot.reply_to(
                message,
                "Thanks! You're all set. Try asking me something or use /help to see all commands."
            )
            logger.info(f"Contact information saved for user: {message.chat.id}")
    except Exception as e:
        logger.error(f"Error handling contact: {e}")
        bot.reply_to(message, "Sorry, I couldn't save your contact information. Please try again.")

@bot.message_handler(commands=['help'])
def help_command(message: Message):
    """Handle /help command."""
    try:
        help_text = (
            "Here's what I can do:\n\n"
            "/chat - Start a conversation with me\n"
            "/websearch - Search the web\n"
            "/stats - View your usage statistics\n\n"
            "You can also send me images or files for analysis!"
        )
        bot.reply_to(message, help_text)
        logger.info(f"Help command processed for user: {message.chat.id}")
    except Exception as e:
        logger.error(f"Error in help command: {e}")
        bot.reply_to(message, "Sorry, I couldn't process the help command. Please try again.")

@bot.message_handler(commands=['websearch'])
def web_search_command(message: Message):
    """Handle /websearch command."""
    try:
        bot.reply_to(message, "What would you like to search for?")
        bot.register_next_step_handler(message, perform_web_search)
        logger.info(f"Web search initiated for user: {message.chat.id}")
    except Exception as e:
        logger.error(f"Error in websearch command: {e}")
        bot.reply_to(message, "Sorry, I couldn't process the web search command. Please try again.")

def perform_web_search(message: Message):
    """Perform web search based on user query."""
    try:
        summary, links = WebSearcher.search_and_summarize(message.text)
        response = f"Here's what I found:\n\n{summary}\n\nSources:\n"
        response += "\n".join([f"• {link}" for link in links[:5]])
        bot.reply_to(message, response)
        logger.info(f"Web search completed for user: {message.chat.id}")
    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        bot.reply_to(message, "Sorry, I couldn't complete the web search. Please try again.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message: Message):
    """Handle photo messages."""
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        with open("temp_image.jpg", "wb") as temp_file:
            temp_file.write(downloaded_file)
        
        analysis = FileAnalyzer.analyze_image("temp_image.jpg")
        FileAnalyzer.save_file_metadata(
            message.chat.id,
            file_info.file_path,
            analysis[:100]
        )
        
        bot.reply_to(message, f"Image Analysis:\n\n{analysis}")
        os.remove("temp_image.jpg")
        logger.info(f"Photo processed for user: {message.chat.id}")
    except Exception as e:
        logger.error(f"Error handling photo: {e}")
        bot.reply_to(message, "Sorry, I couldn't process this image. Please try again.")

@bot.message_handler(content_types=['document'])
def handle_document(message: Message):
    """Handle document messages."""
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        file_path = f"temp_{message.document.file_name}"
        with open(file_path, "wb") as temp_file:
            temp_file.write(downloaded_file)
        
        if file_path.lower().endswith('.pdf'):
            analysis = FileAnalyzer.analyze_pdf(file_path)
        else:
            analysis = "Sorry, I can only analyze PDF documents at the moment."
        
        FileAnalyzer.save_file_metadata(
            message.chat.id,
            message.document.file_name,
            analysis[:100]
        )
        
        bot.reply_to(message, f"Document Analysis:\n\n{analysis}")
        os.remove(file_path)
        logger.info(f"Document processed for user: {message.chat.id}")
    except Exception as e:
        logger.error(f"Error handling document: {e}")
        bot.reply_to(message, "Sorry, I couldn't process this document. Please try again.")

@bot.message_handler(commands=['stats'])
def show_stats(message: Message):
    """Show user statistics."""
    try:
        user = db.users.find_one({'chat_id': message.chat.id})
        if user:
            stats = (
                f"Your Statistics:\n\n"
                f"• Joined: {user['joined_at'].strftime('%Y-%m-%d')}\n"
                f"• Total messages: {user['total_messages']}\n"
                f"• Last active: {user['last_active'].strftime('%Y-%m-%d %H:%M')}"
            )
            bot.reply_to(message, stats)
            logger.info(f"Stats displayed for user: {message.chat.id}")
        else:
            bot.reply_to(message, "Sorry, I couldn't find your statistics. Try using /start first.")
    except Exception as e:
        logger.error(f"Error showing stats: {e}")
        bot.reply_to(message, "Sorry, I couldn't retrieve your statistics. Please try again.")

@bot.message_handler(func=lambda message: True)
def handle_message(message: Message):
    """Handle all other messages."""
    try:
        response = model.generate_content(message.text)
        ChatHistory.save_message(message.chat.id, message.text, response.text)
        bot.reply_to(message, response.text)
        logger.info(f"Message handled for user: {message.chat.id}")
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        bot.reply_to(message, "Sorry, I couldn't process your message. Please try again.")

# Start the bot
if __name__ == '__main__':
    try:
        logger.info("Bot started...")
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise