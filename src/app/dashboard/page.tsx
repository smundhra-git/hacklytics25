'use client';

import React, { useEffect, useState, useRef } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Image from 'next/image';
import { 
  HomeIcon, SendIcon, FileTextIcon, BrainCircuitIcon,
  MailIcon, FileSignatureIcon, RocketIcon, SettingsIcon,
  LogOutIcon, PlusIcon, SearchIcon, BellIcon, Sparkles,
  MessageSquareIcon, ChevronDownIcon, ImageIcon,
  FileIcon
} from 'lucide-react';
import { AuthService } from '@/services/auth.service';

interface UserInfo {
  email: string;
  full_name: string;
  is_active: boolean;
}

interface Message {
  text: string;
  isUser: boolean;
  timestamp: Date;
  status?: 'sending' | 'sent' | 'error';
}

export default function DashboardPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedSection, setSelectedSection] = useState('Chat');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showAttachMenu, setShowAttachMenu] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const token = searchParams.get('token');
    if (token) {
      // Store the token
      AuthService.setToken(token);
      // Clean up URL
      window.history.replaceState({}, '', '/dashboard');
    }

    // Check if we have a token, if not redirect to login
    if (!AuthService.isAuthenticated()) {
      AuthService.logout();
      return;
    }

    const fetchUserInfo = async () => {
      try {
        const info = await AuthService.getUserInfo();
        if (!info) {
          throw new Error('Failed to get user info');
        }
        
        setUserInfo(info);
        setMessages([{
          text: `Welcome back, ${info.full_name}! How can I assist you today?`,
          isUser: false,
          timestamp: new Date(),
          status: 'sent'
        }]);
      } catch (error) {
        console.error('Error fetching user info:', error);
        AuthService.logout();
      } finally {
        setLoading(false);
      }
    };

    fetchUserInfo();
  }, [router, searchParams]);

  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;

    const userMessage: Message = {
      text: currentMessage,
      isUser: true,
      timestamp: new Date(),
      status: 'sending'
    };
    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsTyping(true);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${AuthService.getToken()}`
        },
        body: JSON.stringify({ message: currentMessage })
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      
      setMessages(prev => {
        const updated = [...prev];
        const userMessageIndex = updated.length - 1;
        updated[userMessageIndex] = { ...updated[userMessageIndex], status: 'sent' };
        return updated;
      });

      const assistantMessage: Message = {
        text: data.response,
        isUser: false,
        timestamp: new Date(),
        status: 'sent'
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => {
        const updated = [...prev];
        const userMessageIndex = updated.length - 1;
        updated[userMessageIndex] = { ...updated[userMessageIndex], status: 'error' };
        return updated;
      });
    } finally {
      setIsTyping(false);
    }
  };

  const handleLogout = async () => {
    AuthService.logout();
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-black"></div>
          <p className="text-gray-600">Loading your workspace...</p>
        </div>
      </div>
    );
  }

  const navigationItems = [
    { id: 'chat', label: 'Chat', icon: MessageSquareIcon },
    { id: 'automations', label: 'Automations', icon: BrainCircuitIcon, 
      subItems: [
        { id: 'email', label: 'Email Automation', icon: MailIcon },
        { id: 'docusign', label: 'DocuSign', icon: FileSignatureIcon },
        { id: 'custom', label: 'Custom Workflows', icon: RocketIcon },
      ]
    },
  ];

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className="w-80 bg-[#B2EBF2] p-6 flex flex-col rounded-r-[32px] shadow-lg">
        <div className="flex items-center gap-3 mb-8">
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-[#B2EBF2] to-[#B7BFFF] rounded-full blur opacity-25 group-hover:opacity-75 transition duration-200"></div>
            <div className="relative flex items-center">
              <Image 
                src="/image.png"
                alt="Logo"
                width={40}
                height={40}
                priority
                className="object-contain transform group-hover:scale-105 transition-transform duration-200"
              />
              <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-black to-gray-600 ml-2">
                Adam AI
              </span>
            </div>
          </div>
        </div>

        {/* Search Bar */}
        <div className="relative mb-8">
          <input
            type="text"
            placeholder="Search..."
            className="w-full p-3 pl-10 rounded-xl bg-white/50 border border-transparent focus:border-black/20 focus:bg-white transition-all duration-300"
          />
          <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500" size={18} />
        </div>

        {/* Navigation */}
        <div className="flex-1 space-y-2">
          {navigationItems.map((item) => (
            <div key={item.id} className="space-y-2">
              <button
                onClick={() => setSelectedSection(item.label)}
                className={`flex items-center gap-3 w-full p-3 rounded-xl transition-all duration-200 ${
                  selectedSection === item.label
                    ? 'bg-white text-black shadow-md'
                    : 'hover:bg-white/50 text-gray-700'
                }`}
              >
                <item.icon size={20} />
                <span className="font-medium">{item.label}</span>
                {item.subItems && (
                  <ChevronDownIcon size={16} className="ml-auto" />
                )}
              </button>
              {item.subItems && (
                <div className="ml-4 space-y-1">
                  {item.subItems.map((subItem) => (
                    <button
                      key={subItem.id}
                      onClick={() => setSelectedSection(subItem.label)}
                      className={`flex items-center gap-3 w-full p-2 rounded-lg transition-all duration-200 ${
                        selectedSection === subItem.label
                          ? 'bg-white/80 text-black'
                          : 'hover:bg-white/30 text-gray-600'
                      }`}
                    >
                      <subItem.icon size={18} />
                      <span className="font-medium text-sm">{subItem.label}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* User Profile Section */}
        <div className="mt-auto pt-4 border-t border-black/10">
          <div className="flex items-center justify-between p-2">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-10 h-10 rounded-full bg-white/80 flex items-center justify-center text-lg font-semibold text-gray-700">
                  {userInfo?.full_name?.[0]?.toUpperCase()}
                </div>
                <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
              </div>
              <div className="flex flex-col">
                <span className="font-medium text-sm">{userInfo?.full_name}</span>
                <span className="text-xs text-gray-600">{userInfo?.email}</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button 
                onClick={handleLogout}
                className="p-2 hover:bg-white/50 rounded-xl transition-colors"
                title="Logout"
              >
                <LogOutIcon size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="h-16 border-b border-gray-200 flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-semibold">{selectedSection}</h1>
            {selectedSection !== 'Chat' && (
              <button className="flex items-center gap-2 px-4 py-2 bg-black text-white rounded-xl hover:bg-gray-800 transition-all duration-300">
                <PlusIcon size={18} />
                <span>New {selectedSection}</span>
              </button>
            )}
          </div>
          <div className="flex items-center gap-4">
            <button className="p-2 hover:bg-gray-100 rounded-full transition-colors relative">
              <BellIcon size={20} />
              <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>
            <button className="p-2 hover:bg-gray-100 rounded-full transition-colors">
              <SettingsIcon size={20} />
            </button>
          </div>
        </div>

        {/* Dynamic Content Area */}
        <div className="flex-1 p-6 overflow-y-auto">
          {selectedSection === 'Chat' ? (
            // Chat Interface
            <div className="h-full flex flex-col">
              <div className="flex-1 overflow-y-auto space-y-6">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`mb-6 flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    {!message.isUser && (
                      <div className="w-8 h-8 rounded-full bg-[#B2EBF2] flex items-center justify-center mr-3">
                        <Image 
                          src="/image.png"
                          alt="Assistant"
                          width={20}
                          height={20}
                          className="object-contain"
                        />
                      </div>
                    )}
                    <div
                      className={`max-w-[70%] p-4 ${
                        message.isUser
                          ? 'bg-black text-white rounded-[24px] rounded-tr-none'
                          : 'bg-[#E8EAF6] rounded-[24px] rounded-tl-none'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{message.text}</p>
                      <div className={`mt-2 text-xs ${message.isUser ? 'text-white/70' : 'text-gray-500'} flex items-center gap-2`}>
                        {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        {message.isUser && message.status && (
                          <span className="flex items-center gap-1">
                            {message.status === 'sending' && '• Sending...'}
                            {message.status === 'sent' && '• Sent'}
                            {message.status === 'error' && '• Failed to send'}
                          </span>
                        )}
                      </div>
                    </div>
                    {message.isUser && (
                      <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center ml-3">
                        {userInfo?.full_name?.[0]?.toUpperCase()}
                      </div>
                    )}
                  </div>
                ))}
                {isTyping && (
                  <div className="flex items-center gap-2 text-gray-500">
                    <div className="w-8 h-8 rounded-full bg-[#B2EBF2] flex items-center justify-center">
                      <Image 
                        src="/image.png"
                        alt="Assistant"
                        width={20}
                        height={20}
                        className="object-contain"
                      />
                    </div>
                    <div className="bg-[#E8EAF6] px-4 py-2 rounded-[24px] rounded-tl-none">
                      <div className="flex gap-1">
                        <span className="animate-bounce">•</span>
                        <span className="animate-bounce" style={{ animationDelay: '0.2s' }}>•</span>
                        <span className="animate-bounce" style={{ animationDelay: '0.4s' }}>•</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
              
              {/* Chat Input */}
              <div className="mt-6">
                <div className="flex gap-4 items-center">
                  <div className="relative">
                    <button
                      onClick={() => setShowAttachMenu(!showAttachMenu)}
                      className="p-3 hover:bg-gray-100 rounded-full transition-colors"
                    >
                      <PlusIcon size={24} />
                    </button>
                    {showAttachMenu && (
                      <div className="absolute bottom-full left-0 mb-2 bg-white rounded-xl shadow-lg border border-gray-200 p-2 min-w-[160px]">
                        <button className="flex items-center gap-2 w-full p-2 hover:bg-gray-100 rounded-lg text-left">
                          <ImageIcon size={18} />
                          <span>Image</span>
                        </button>
                        <button className="flex items-center gap-2 w-full p-2 hover:bg-gray-100 rounded-lg text-left">
                          <FileIcon size={18} />
                          <span>Document</span>
                        </button>
                      </div>
                    )}
                  </div>
                  <input
                    type="text"
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="How can I assist you today?"
                    className="flex-1 p-4 rounded-full bg-gray-100 border-0 focus:outline-none focus:ring-2 focus:ring-black/10"
                  />
                  <button
                    onClick={handleSendMessage}
                    className="p-4 bg-black text-white rounded-full hover:bg-gray-800 transition-all duration-300"
                  >
                    <SendIcon size={20} />
                  </button>
                </div>
              </div>
            </div>
          ) : (
            // Automation Interface
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {selectedSection === 'Email Automation' && (
                <>
                  <div className="p-6 bg-white rounded-2xl border border-gray-200 hover:shadow-lg transition-all duration-300">
                    <div className="flex items-center justify-between mb-4">
                      <MailIcon className="text-blue-500" size={24} />
                      <Sparkles className="text-yellow-500" size={20} />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">Email Scheduler</h3>
                    <p className="text-gray-600 text-sm">Schedule and automate your email campaigns</p>
                  </div>
                </>
              )}
              
              {selectedSection === 'DocuSign' && (
                <>
                  <div className="p-6 bg-white rounded-2xl border border-gray-200 hover:shadow-lg transition-all duration-300">
                    <div className="flex items-center justify-between mb-4">
                      <FileSignatureIcon className="text-green-500" size={24} />
                      <Sparkles className="text-yellow-500" size={20} />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">Document Signing</h3>
                    <p className="text-gray-600 text-sm">Automate your document signing workflow</p>
                  </div>
                </>
              )}
              
              {selectedSection === 'Custom Workflows' && (
                <>
                  <div className="p-6 bg-white rounded-2xl border border-gray-200 hover:shadow-lg transition-all duration-300">
                    <div className="flex items-center justify-between mb-4">
                      <RocketIcon className="text-purple-500" size={24} />
                      <Sparkles className="text-yellow-500" size={20} />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">Custom Automation</h3>
                    <p className="text-gray-600 text-sm">Create your own custom automation workflow</p>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 