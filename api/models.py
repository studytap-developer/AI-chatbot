from sqlalchemy import Boolean, Column, DateTime, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from api.database import Base  # Import the Base class from database.py
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True)  # Firebase UID

    name = Column(String(255), index=True)           # ✅ Added length
    email = Column(String(255), unique=True, index=True)  # ✅ Added length
    is_subscribed = Column(Boolean, default=False)
    subscription_expiry = Column(DateTime, nullable=True)
    chat_history = relationship("ChatHistory", back_populates="user")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    user_message = Column(Text)
    question = Column(String)
    answer = Column(String)
    bot_response = Column(Text)
    timestamp = Column(Integer)

    user = relationship("User", back_populates="chat_history")
