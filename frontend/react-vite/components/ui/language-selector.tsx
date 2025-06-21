"use client";

import { useState } from "react";
import { ChevronDown, Globe } from "lucide-react";

// Language options with Kokoro codes
const LANGUAGES = [
  { code: 'a', name: 'English (US)', flag: '🇺🇸' },
  { code: 'b', name: 'English (UK)', flag: '🇬🇧' },
  { code: 'e', name: 'Spanish', flag: '🇪🇸' },
  { code: 'i', name: 'Italian', flag: '🇮🇹' },
  { code: 'f', name: 'French', flag: '🇫🇷' },
  { code: 'p', name: 'Portuguese', flag: '🇵🇹' },
  { code: 'j', name: 'Japanese', flag: '🇯🇵' },
  { code: 'z', name: 'Chinese', flag: '🇨🇳' },
  { code: 'h', name: 'Hindi', flag: '🇮🇳' },
];

export function LanguageSelector() {
  const [selectedLanguage, setSelectedLanguage] = useState(LANGUAGES[0]);
  const [isOpen, setIsOpen] = useState(false);

  const handleLanguageSelect = async (language: typeof LANGUAGES[0]) => {
    setSelectedLanguage(language);
    setIsOpen(false);
    
    // Send language to backend
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const fullUrl = `${apiUrl}/api/set-language`;
      console.log('Environment variables:', {
        NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
        apiUrl: apiUrl,
        fullUrl: fullUrl
      });
      
      const response = await fetch(fullUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ language: language.code }),
      });
      
      console.log('Response status:', response.status);
      const result = await response.json();
      console.log('Response:', result);
    } catch (error) {
      console.error('Failed to set language:', error);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-background/80 backdrop-blur-sm border border-border rounded-lg hover:bg-accent/50 transition-colors"
      >
        <Globe className="w-4 h-4" />
        <span className="text-lg">{selectedLanguage.flag}</span>
        <span className="text-sm font-medium">{selectedLanguage.name}</span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full mt-2 left-0 w-64 bg-background/95 backdrop-blur-sm border border-border rounded-lg shadow-lg overflow-hidden z-50">
          {LANGUAGES.map((language) => (
            <button
              key={language.code}
              onClick={() => handleLanguageSelect(language)}
              className={`w-full flex items-center gap-3 px-4 py-3 hover:bg-accent/50 transition-colors text-left ${
                selectedLanguage.code === language.code ? 'bg-accent/30' : ''
              }`}
            >
              <span className="text-lg">{language.flag}</span>
              <span className="text-sm font-medium">{language.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}