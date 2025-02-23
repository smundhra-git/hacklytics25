import { authConfig } from '@/config/auth';
import Cookies from 'js-cookie';

export class AuthService {
  private static readonly TOKEN_KEY = 'auth_token';
  private static readonly API_URL = process.env.NEXT_PUBLIC_API_URL;

  static getToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(this.TOKEN_KEY);
    }
    return null;
  }

  static setToken(token: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem(this.TOKEN_KEY, token);
    }
  }

  static removeToken(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(this.TOKEN_KEY);
    }
  }

  static isAuthenticated(): boolean {
    return !!this.getToken();
  }

  static async getUserInfo() {
    const token = this.getToken();
    if (!token) return null;

    try {
      const response = await fetch(`${this.API_URL}/api/v1/auth/me`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch user info');
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching user info:', error);
      this.logout();
      return null;
    }
  }

  static initiateGoogleLogin(): void {
    if (typeof window !== 'undefined') {
      window.location.href = `${this.API_URL}/api/v1/auth/login/google`;
    }
  }

  static async handleAuthCallback(token: string): Promise<void> {
    this.setToken(token);
    
    // Get return path or default to dashboard
    const returnPath = localStorage.getItem('returnPath') || authConfig.routes.dashboard;
    localStorage.removeItem('returnPath'); // Clean up
    
    window.location.href = returnPath;
  }

  static logout(): void {
    this.removeToken();
    if (typeof window !== 'undefined') {
      window.location.replace('/login');
    }
  }
} 