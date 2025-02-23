'use client';

import React from 'react';
import Link from 'next/link';
import { FolderIcon } from 'lucide-react';
import Image from 'next/image';
import { Button } from '@/components/ui/button';

const navigationItems = [
  { href: "/", label: "Home" },
  { href: "/features", label: "Features" },
  { href: "/about", label: "About" },
  { href: "/blog", label: "Blog" }
];

const avatarItems = [1, 2, 3];

export default function Home() {
  return (
    <main className="min-h-screen bg-white">
      <div className="h-screen p-4 flex flex-col">
        {/* Hero Section with Navigation */}
        <div className="bg-[#C3EFF6] rounded-[32px] p-8 h-[75vh] flex flex-col">
          {/* Navigation */}
          <nav className="flex items-center justify-between mb-12 bg-white/10 backdrop-blur-sm px-8 py-4 rounded-2xl border border-white/20">
            <div className="flex items-center gap-3">
              <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-[#C3EFF6] to-[#B7BFFF] rounded-full blur opacity-25 group-hover:opacity-75 transition duration-200"></div>
                <div className="relative flex items-center">
                  <Image 
                    src="/image.png"
                    alt="Logo"
                    width={36}
                    height={36}
                    priority
                    className="object-contain"
                  />
                  <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-black to-gray-600 ml-2">
                    Adam AI
                  </span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-12">
              {navigationItems.map(({ href, label }) => (
                <Link 
                  key={href} 
                  href={href} 
                  className="relative group py-2"
                >
                  <span className="relative z-10 text-gray-800 font-medium group-hover:text-black transition-colors duration-200">
                    {label}
                  </span>
                  <span className="absolute bottom-0 left-0 w-full h-0.5 bg-black transform origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-300"></span>
                </Link>
              ))}
            </div>

            <div className="flex items-center gap-4">
              <Link 
                href="/contact"
                className="text-gray-600 hover:text-black transition-colors duration-200 text-sm font-medium"
              >
                Contact
              </Link>
              <div className="h-4 w-px bg-gray-300"></div>
              <Link 
                href="/login"
                className="relative group"
              >
                <span className="absolute -inset-1 bg-black rounded-xl blur opacity-0 group-hover:opacity-10 transition duration-200"></span>
                <span className="relative px-6 py-2.5 border-2 border-black rounded-xl hover:bg-black hover:text-white transition-all duration-300 flex items-center gap-2">
                  Get Started
                  <svg 
                    className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-200"
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M13 7l5 5m0 0l-5 5m5-5H6"
                    />
                  </svg>
                </span>
              </Link>
            </div>
          </nav>

          {/* Hero Content */}
          <div className="flex justify-between items-center flex-1">
            <div className="max-w-2xl">
              <h1 className="text-[64px] font-bold leading-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-black via-gray-700 to-gray-900">
                Your Personal AI<br />
                Assistant for Modern<br />
                Productivity
              </h1>
              <p className="text-xl text-gray-700 mb-8 max-w-lg leading-relaxed">
                Experience the future of work with Adam AI. Intelligent task management, 
                smart automation, and context-aware assistance to help you achieve more.
              </p>
              <div className="flex gap-4">
                <Button 
                  asChild
                  className="px-8 py-6 bg-black text-white rounded-xl hover:bg-gray-800 transition-all duration-300 transform hover:scale-105 shadow-lg"
                >
                  <Link href="/login">Try Adam Free</Link>
                </Button>
                <Button 
                  asChild
                  variant="outline"
                  className="px-8 py-6 border-2 border-black rounded-xl hover:bg-black hover:text-white transition-all duration-300 transform hover:scale-105 group"
                >
                  <Link href="/demo">
                    <span className="flex items-center gap-2">
                      Watch Demo
                      <svg
                        className="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 5l7 7-7 7"
                        />
                      </svg>
                    </span>
                  </Link>
                </Button>
              </div>
            </div>
            <div className="flex-shrink-0 relative">
              <div className="absolute -top-10 -left-10 w-72 h-72 bg-blue-200 rounded-full filter blur-3xl opacity-30 animate-pulse"></div>
              <div className="absolute -bottom-10 -right-10 w-72 h-72 bg-purple-200 rounded-full filter blur-3xl opacity-30 animate-pulse"></div>
              <FolderIcon className="w-[400px] h-[400px] text-black relative z-10" />
            </div>
          </div>
        </div>

        {/* Stats Section */}
        <div className="grid grid-cols-4 gap-6 h-[25vh] mt-4">
          <div className="bg-[#B7BFFF] rounded-[24px] p-6 flex flex-col justify-between relative overflow-hidden group hover:shadow-lg transition-all duration-300">
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full transform translate-x-8 -translate-y-8"></div>
            <div>
              <h3 className="text-xl font-semibold mb-2">Active Users</h3>
              <p className="text-sm text-gray-700">Growing community of professionals</p>
            </div>
            <div className="flex items-end justify-between">
              <span className="text-4xl font-bold">1000+</span>
              <div className="flex -space-x-2">
                {avatarItems.map((i) => (
                  <div
                    key={i}
                    className="w-8 h-8 rounded-full bg-white/80 border-2 border-[#B7BFFF] transform transition-transform group-hover:translate-y-[-2px]"
                    style={{ transitionDelay: `${i * 100}ms` }}
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="bg-[#B7BFFF] rounded-[24px] p-6 flex flex-col justify-between relative overflow-hidden group hover:shadow-lg transition-all duration-300">
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full transform translate-x-8 -translate-y-8"></div>
            <div>
              <h3 className="text-xl font-semibold mb-2">Tasks Automated</h3>
              <p className="text-sm text-gray-700">Daily productivity enhanced</p>
            </div>
            <div className="flex items-end justify-between">
              <span className="text-4xl font-bold">10M+</span>
              <FolderIcon className="w-12 h-12 text-white/80 transform transition-transform group-hover:rotate-12" />
            </div>
          </div>

          <div className="bg-[#B7BFFF] rounded-[24px] p-6 flex flex-col justify-between relative overflow-hidden group hover:shadow-lg transition-all duration-300">
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full transform translate-x-8 -translate-y-8"></div>
            <div>
              <h3 className="text-xl font-semibold mb-2">Time Saved</h3>
              <p className="text-sm text-gray-700">Hours returned to you</p>
            </div>
            <div className="flex items-end justify-between">
              <span className="text-4xl font-bold">2.5M+</span>
              <div className="w-12 h-12 rounded-full bg-white/80 flex items-center justify-center text-[#B7BFFF] text-2xl transform transition-transform group-hover:scale-110">
                ⏰
              </div>
            </div>
          </div>

          <div className="border border-black rounded-[24px] flex flex-col justify-center items-center p-6 group hover:bg-black hover:text-white transition-all duration-500 hover:shadow-lg">
            <Link
              href="/request-demo"
              className="text-center flex flex-col items-center"
            >
              <span className="text-2xl font-semibold mb-2">See Adam in Action</span>
              <span className="text-sm text-gray-600 group-hover:text-gray-300 mb-4">Book a personalized demo</span>
              <div className="w-10 h-10 rounded-full border-2 border-current flex items-center justify-center transform transition-transform group-hover:rotate-90">
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 12h14M12 5l7 7-7 7"
                  />
                </svg>
              </div>
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}
