import React, { useState, useRef, useEffect } from 'react';
import { Send, Sun, Moon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import '../styles/globalStyles.css'

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'therapist';
}

const AnimatedHeader: React.FC<{ isLandingPage: boolean; darkMode: boolean }> = ({ isLandingPage, darkMode }) => {
  const longTitle = "CAPT: Conversational Agent for Psychedelic-assisted Therapy";
  const shortTitle = "CAPT";

  return (
    <div className="relative h-10 overflow-hidden">
      <AnimatePresence mode="wait">
        <motion.h1
          key={isLandingPage ? 'long' : 'short'}
          initial={{ 
            opacity: 1,
            y: isLandingPage ? 40 : -40
          }}
          animate={{ 
            opacity: 1,
            y: 0 
          }}
          exit={{ 
            opacity: 1,
            y: isLandingPage ? -40 : 40
          }}
          transition={{ 
            duration: 0.25,
            ease: "easeInOut"
          }}
          className={`font-inter font-bold tracking-tight absolute w-full text-2xl md:text-3xl ${
            darkMode ? 'text-gray-100' : 'text-gray-800'
          }`}
        >
          {isLandingPage ? longTitle : shortTitle}
        </motion.h1>
      </AnimatePresence>
    </div>
  );
};

const ThinkingAnimation: React.FC = () => (
  <div className="flex space-x-1 p-2">
    <motion.div
      className="w-2 h-2 bg-gray-500 rounded-full"
      animate={{ y: [0, -5, 0] }}
      transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse" }}
    />
    <motion.div
      className="w-2 h-2 bg-gray-500 rounded-full"
      animate={{ y: [0, -5, 0] }}
      transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse", delay: 0.1 }}
    />
    <motion.div
      className="w-2 h-2 bg-gray-500 rounded-full"
      animate={{ y: [0, -5, 0] }}
      transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse", delay: 0.2 }}
    />
  </div>
);

const TherapistChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [darkMode, setDarkMode] = useState(true);
  const [isLandingPage, setIsLandingPage] = useState(true);
  const [isThinking, setIsThinking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      const newMessage: Message = {
        id: Date.now(),
        text: input.trim(),
        sender: 'user',
      };
      setMessages([...messages, newMessage]);
      setInput('');
      setIsLandingPage(false);
      setIsThinking(true);
      setTimeout(() => {
        const therapistResponse: Message = {
          id: Date.now() + 1,
          text: "Thank you for sharing. I'm here to listen and support you. How does this situation make you feel emotionally?",
          sender: 'therapist',
        };
        setMessages(prev => [...prev, therapistResponse]);
        setIsThinking(false);
      }, 2000);
    }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div className={`flex flex-col h-screen w-full transition-colors duration-300 font-inter ${
      darkMode ? 'bg-gray-900 text-gray-100' : 'bg-light-grey text-gray-800'
    }`}>
      <header className={`p-4 text-center transition-colors duration-300 shadow-md ${
        darkMode ? 'bg-gray-800' : 'bg-soft-blue'
      }`}>
        <AnimatedHeader isLandingPage={isLandingPage} darkMode={darkMode} />
        <button 
          onClick={toggleDarkMode} 
          className="absolute top-4 right-4 p-2 rounded-full hover:bg-opacity-20 hover:bg-white transition-colors duration-300 shadow-sm"
        >
          {darkMode ? <Sun size={24} color="#F3E6D0" /> : <Moon size={24} color="#6B8E8E" />}
        </button>
      </header>
      
      <AnimatePresence>
        {isLandingPage ? (
          <motion.div
            key="landing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex-grow flex flex-col items-center justify-center p-4 sm:p-8 text-center"
          >
            <h2 className="font-semibold tracking-tight mb-4 shadow-sm">Welcome to CAPT—your partner in safe & effective psychedelic-assisted therapy.</h2>
            <p className="mb-8 font-normal leading-relaxed">Feel free to share your thoughts or concerns with me. I'm here to listen and support you.</p>
            <div className="w-full max-w-md space-y-4">
              <form onSubmit={handleSubmit}>
                <div className='flex flex-col gap-4'>
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message here..."
                    className={`w-full p-4 rounded-xl shadow-lg focus:outline-none focus:ring-2 focus:ring-gentle-green transition-all duration-300 font-normal ${
                      darkMode
                        ? 'bg-gray-700 text-gray-100 placeholder-gray-400 hover:bg-gray-600'
                        : 'bg-white text-gray-800 placeholder-gray-500 hover:bg-gray-50'
                    } hover:shadow-xl`}
                  />
                  <button
                    type="submit"
                    className={`w-full p-4 rounded-xl shadow-lg focus:outline-none focus:ring-2 focus:ring-gentle-green transition-all duration-300 transform hover:scale-102 flex items-center justify-center space-x-3 font-medium ${
                      darkMode
                        ? 'bg-soft-blue text-gray-800 hover:bg-opacity-90'
                        : 'bg-soft-blue text-gray-800 hover:bg-opacity-90'
                    } hover:shadow-xl`}
                  >
                    <span>Start Chatting</span>
                    <Send size={20} className="ml-2" />
                  </button>
                </div>
              </form>
            </div>
          </motion.div>
        ) : (
          <motion.div 
            key="chat"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex-grow flex flex-col"
          >
            <div className="flex-grow overflow-auto p-4">
              <AnimatePresence>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 50 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -50 }}
                    transition={{ duration: 0.5 }}
                    className={`mb-4 ${message.sender === 'user' ? 'text-right' : 'text-left'}`}
                  >
                    <div
                      className={`inline-block p-3 rounded-lg max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg shadow-md hover:shadow-lg transition-shadow duration-300 font-normal leading-relaxed ${
                        message.sender === 'user'
                          ? 'bg-soft-blue text-gray-800'
                          : `bg-warm-beige text-gray-800`
                      }`}
                    >
                      {message.text}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              {isThinking && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="mb-4 text-left"
                >
                  <div className={`inline-block p-3 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-warm-beige'}`}>
                    <ThinkingAnimation />
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <form onSubmit={handleSubmit} className={`p-4 transition-colors duration-300 shadow-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type your message here..."
                  className={`flex-grow p-4 rounded-xl shadow-md focus:outline-none focus:ring-2 focus:ring-gentle-green transition-all duration-300 font-normal ${
                    darkMode 
                      ? 'bg-gray-700 text-gray-100 placeholder-gray-400 hover:bg-gray-600' 
                      : 'bg-light-grey text-gray-800 placeholder-gray-500 hover:bg-gray-100'
                  } hover:shadow-lg`}
                />
                <button
                  type="submit"
                  className={`p-4 rounded-xl shadow-md focus:outline-none focus:ring-2 focus:ring-gentle-green transition-all duration-300 transform hover:scale-105 ${
                    darkMode 
                      ? 'bg-soft-blue text-gray-800 hover:bg-opacity-90' 
                      : 'bg-soft-blue text-gray-800 hover:bg-opacity-90'
                  } hover:shadow-lg`}
                >
                  <Send size={24} />
                </button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default TherapistChat;