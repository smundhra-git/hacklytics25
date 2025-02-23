'use client';

import React from 'react';
import Screen from '@/components/Screen';
import GoogleLoginButton from '@/components/GoogleLoginButton';
import Image from 'next/image';
import { ArrowLeft, Sparkles, Bot, Shield, Zap } from 'lucide-react';
import Link from 'next/link';

export default function LoginPage() {
  return (
    <Screen>
      <div className="min-h-screen flex bg-white relative overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-0 left-0 w-[500px] h-[500px] bg-[#B2EBF2] rounded-full filter blur-3xl opacity-20 animate-blob"></div>
          <div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-[#B7BFFF] rounded-full filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-purple-200 rounded-full filter blur-3xl opacity-10 animate-blob animation-delay-4000"></div>
        </div>

        {/* Left Section - Features */}
        <div className="hidden lg:flex w-1/2 bg-gradient-to-br from-[#C3EFF6] to-[#B7BFFF] p-12 relative">
          <div className="relative z-10 flex flex-col justify-between w-full">
            <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity w-fit">
              <ArrowLeft className="w-5 h-5" />
              <span className="font-medium">Back to Home</span>
            </Link>

            <div className="space-y-8">
              <div className="flex items-center gap-4">
                <Image 
                  src="/image.png"
                  alt="Adam AI Logo"
                  width={48}
                  height={48}
                  className="object-contain"
                  priority
                />
                <h1 className="text-3xl font-bold">Adam AI</h1>
              </div>
              
              <div className="space-y-6">
                <h2 className="text-4xl font-bold leading-tight">
                  Your Intelligent<br />Assistant Awaits
                </h2>
                <p className="text-lg text-gray-700 max-w-md">
                  Join thousands of professionals who trust Adam AI to enhance their productivity and streamline their workflow.
                </p>
              </div>

              <div className="grid gap-6">
                <div className="flex items-center gap-4 bg-white/80 p-4 rounded-2xl backdrop-blur-sm">
                  <div className="w-12 h-12 rounded-xl bg-black flex items-center justify-center text-white">
                    <Bot className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="font-semibold">AI-Powered Assistant</h3>
                    <p className="text-sm text-gray-600">Smart responses and intelligent automation</p>
                  </div>
                </div>

                <div className="flex items-center gap-4 bg-white/80 p-4 rounded-2xl backdrop-blur-sm">
                  <div className="w-12 h-12 rounded-xl bg-black flex items-center justify-center text-white">
                    <Shield className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="font-semibold">Secure & Private</h3>
                    <p className="text-sm text-gray-600">Your data is encrypted and protected</p>
                  </div>
                </div>

                <div className="flex items-center gap-4 bg-white/80 p-4 rounded-2xl backdrop-blur-sm">
                  <div className="w-12 h-12 rounded-xl bg-black flex items-center justify-center text-white">
                    <Zap className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="font-semibold">Lightning Fast</h3>
                    <p className="text-sm text-gray-600">Instant responses and quick actions</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-600">Trusted by industry leaders</span>
              <div className="flex -space-x-3">
                {[1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="w-8 h-8 rounded-full border-2 border-white bg-gray-200"
                  />
                ))}
                <div className="w-8 h-8 rounded-full border-2 border-white bg-black text-white text-xs flex items-center justify-center">
                  +2k
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Section - Login */}
        <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
          <div className="w-full max-w-md space-y-8 relative">
            <div className="text-center relative">
              <div className="flex justify-center mb-2">
                <Sparkles className="w-6 h-6 text-yellow-500 animate-pulse" />
              </div>
              <h2 className="text-2xl font-bold mb-2">Welcome Back</h2>
              <p className="text-gray-600">Sign in to continue to Adam AI</p>
            </div>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-white text-gray-500">Continue with</span>
              </div>
            </div>

            <div className="space-y-6">
              <GoogleLoginButton />
              
              <div className="text-center space-y-4">
                <p className="text-sm text-gray-500">
                  By continuing, you agree to our{' '}
                  <Link href="/terms" className="underline hover:text-black">Terms of Service</Link>
                  {' '}and{' '}
                  <Link href="/privacy" className="underline hover:text-black">Privacy Policy</Link>
                </p>
                
                <div className="flex justify-center gap-8 text-sm text-gray-500">
                  <Link href="/help" className="hover:text-black">Need help?</Link>
                  <Link href="/contact" className="hover:text-black">Contact us</Link>
                </div>
              </div>
            </div>

            {/* Decorative Elements */}
            <div className="absolute top-0 right-0 w-20 h-20 bg-yellow-100 rounded-full filter blur-xl opacity-60 animate-pulse"></div>
            <div className="absolute bottom-0 left-0 w-20 h-20 bg-purple-100 rounded-full filter blur-xl opacity-60 animate-pulse animation-delay-2000"></div>
          </div>
        </div>
      </div>
    </Screen>
  );
} 