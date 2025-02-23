'use client';

import { useEffect } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { AuthService } from '@/services/auth.service';

export default function AuthCallback() {
  const searchParams = useSearchParams();
  const router = useRouter();

  useEffect(() => {
    const token = searchParams.get('token');
    if (token) {
      // Store the token
      AuthService.setToken(token);
      // Redirect to dashboard
      router.push('/dashboard');
    } else {
      // If no token, redirect to login
      router.push('/login');
    }
  }, [searchParams, router]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-white">
      <div className="flex flex-col items-center gap-4">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-black"></div>
        <p className="text-gray-600">Setting up your workspace...</p>
        <p className="text-sm text-gray-500">You'll be redirected to your dashboard shortly</p>
      </div>
    </div>
  );
} 