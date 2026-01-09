import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import VirtualKeyboard from './VirtualKeyboard';
import History from './History';

function App() {
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [result, setResult] = useState(null);
  const [isParagraph, setIsParagraph] = useState(false);
  const [error, setError] = useState('');
  const [translation, setTranslation] = useState('');
  const [speechLang, setSpeechLang] = useState('hi-IN'); // Default to Hindi
  const [showKeyboard, setShowKeyboard] = useState(false);
  const [keyboardLang, setKeyboardLang] = useState('hindi');
  const [showHistory, setShowHistory] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark';
  });

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const textareaRef = useRef(null);
  const recognitionRef = useRef(null);

  const API_BASE_URL = 'http://localhost:8000';

  // Initialize Web Speech API
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      
      recognitionRef.current.onstart = () => {
        setIsRecording(true);
        setError('');
      };
      
      recognitionRef.current.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        setInputText(transcript);
        setRecordedAudio(null);
        setAudioBlob(null);
      };
      
      recognitionRef.current.onerror = (event) => {
        setError(`Speech recognition error: ${event.error}`);
      };
      
      recognitionRef.current.onend = () => {
        setIsRecording(false);
      };
    }
  }, []);

  // Apply theme to body element
  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Toggle theme
  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'dark' ? 'light' : 'dark');
  };

  // Handle text input change
  const handleInputChange = (e) => {
    setInputText(e.target.value);
    setError('');
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!inputText.trim()) {
      setError('Please enter some text or use the microphone');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);
    setTranslation('');

    try {
      // Step 1: Get prediction (now includes translation and explanation)
      const predictionResponse = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!predictionResponse.ok) {
        const errorData = await predictionResponse.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const predictionData = await predictionResponse.json();
      setResult(predictionData);
      setIsParagraph(predictionData.is_paragraph || false);

      // Translation and explanation are now included in the prediction response
      setTranslation(predictionData.translation);
    } catch (err) {
      setError(err.message || 'An error occurred. Please try again later.');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle microphone recording using Web Speech API
  const handleMicrophoneClick = () => {
    if (!recognitionRef.current) {
      setError('Speech recognition not supported in your browser. Please use Chrome, Edge, or Safari.');
      return;
    }

    if (isRecording) {
      // Stop recording
      recognitionRef.current.stop();
    } else {
      // Start recording
      setError('');
      setRecordedAudio(null);
      setAudioBlob(null);
      
      // Set language for recognition
      const langMap = {
        'hi-IN': 'hi-IN',
        'ta-IN': 'ta-IN',
        'te-IN': 'te-IN',
        'kn-IN': 'kn-IN'
      };
      recognitionRef.current.lang = langMap[speechLang] || 'hi-IN';
      recognitionRef.current.start();
      console.log(`Speech recognition started for language: ${recognitionRef.current.lang}`);
    }
  };

  // Web Speech API handles transcription directly, no backend call needed
  // const handleSendAudio = async () => {
  //   // Commented out: Web Speech API transcription is handled in the recognition.onresult event
  // };

  // Delete recorded audio
  const handleDeleteAudio = () => {
    if (recordedAudio) {
      URL.revokeObjectURL(recordedAudio);
    }
    setRecordedAudio(null);
    setAudioBlob(null);
  };

  // Handle reset
  const handleReset = () => {
    setInputText('');
    setResult(null);
    setError('');
    setTranslation('');
    setIsParagraph(false);
  };

  // Handle virtual keyboard
  const handleKeyboardToggle = (lang) => {
    setKeyboardLang(lang);
    // Only toggle visibility if we're not already showing the keyboard
    if (!showKeyboard) {
      setShowKeyboard(true);
    }
  };

  const handleVirtualKeyPress = (key) => {
    if (key === 'BACKSPACE') {
      setInputText(prev => prev.slice(0, -1));
    } else {
      setInputText(prev => prev + key);
    }

    // Keep focus on textarea
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const handleCloseKeyboard = () => {
    setShowKeyboard(false);
  };

  // Get result color based on label
  const getResultColor = (label) => {
    return label === 'metaphor' ? 'result-metaphor' : 'result-normal';
  };

  return (
    <div className="app-container">
      <div className="main-card">
        <header className="app-header">
          <div className="header-content">
            <div>
              <h1>Multilingual Metaphor Detector and Interpreter</h1>
              <p className="subtitle">AI-powered metaphor detection for Hindi, Tamil, Telugu, and Kannada</p>
            </div>
            <div className="header-buttons">
              <button
                className="theme-toggle-btn"
                onClick={toggleTheme}
                title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
              >
                {theme === 'dark' ? '☀️' : '🌙'}
              </button>
              <button
                className="history-btn"
                onClick={() => setShowHistory(true)}
                title="View History"
              >
                📜 History
              </button>
            </div>
          </div>
        </header>

        <form onSubmit={handleSubmit} className="input-section">
          <div className="input-group">
            <textarea
              ref={textareaRef}
              className="text-input"
              placeholder="Enter text in Hindi, Tamil, Telugu, or Kannada..."
              value={inputText}
              onChange={handleInputChange}
              rows="4"
              disabled={isLoading}
            />

            <div className="keyboard-selector">
              <label>⌨️ Virtual Keyboard:</label>
              <div className="keyboard-buttons">
                <button
                  type="button"
                  className={`keyboard-btn ${showKeyboard && keyboardLang === 'hindi' ? 'active' : ''}`}
                  onClick={() => handleKeyboardToggle('hindi')}
                  disabled={isLoading}
                >
                  हिंदी
                </button>
                <button
                  type="button"
                  className={`keyboard-btn ${showKeyboard && keyboardLang === 'tamil' ? 'active' : ''}`}
                  onClick={() => handleKeyboardToggle('tamil')}
                  disabled={isLoading}
                >
                  தமிழ்
                </button>
                <button
                  type="button"
                  className={`keyboard-btn ${showKeyboard && keyboardLang === 'telugu' ? 'active' : ''}`}
                  onClick={() => handleKeyboardToggle('telugu')}
                  disabled={isLoading}
                >
                  తెలుగు
                </button>
                <button
                  type="button"
                  className={`keyboard-btn ${showKeyboard && keyboardLang === 'kannada' ? 'active' : ''}`}
                  onClick={() => handleKeyboardToggle('kannada')}
                  disabled={isLoading}
                >
                  ಕನ್ನಡ
                </button>
              </div>
            </div>

            <div className="speech-lang-selector">
              <label htmlFor="speech-lang">🎤 Speech Language:</label>
              <select
                id="speech-lang"
                value={speechLang}
                onChange={(e) => setSpeechLang(e.target.value)}
                disabled={isLoading || isRecording}
              >
                <option value="hi-IN">Hindi (हिंदी)</option>
                <option value="ta-IN">Tamil (தமிழ்)</option>
                <option value="te-IN">Telugu (తెలుగు)</option>
                <option value="kn-IN">Kannada (ಕನ್ನಡ)</option>
              </select>
            </div>

            <div className="button-group">
              <button
                type="button"
                className={`mic-button ${isRecording ? 'recording' : ''}`}
                onClick={handleMicrophoneClick}
                disabled={isLoading}
                title={isRecording ? 'Stop recording' : 'Start recording'}
              >
                {isRecording ? '⏹️ Stop' : '🎤 Record'}
              </button>

              <button
                type="submit"
                className="submit-button"
                disabled={isLoading || !inputText.trim()}
              >
                {isLoading ? (
                  <>
                    <span className="loading-spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  '🔍 Analyze'
                )}
              </button>

              <button
                type="button"
                className="reset-button"
                onClick={handleReset}
                disabled={isLoading}
              >
                🔄 Reset
              </button>
            </div>
          </div>

          {isRecording && (
            <div className="recording-indicator">
              <span className="recording-dot"></span>
              Recording... Click "Stop" when done
            </div>
          )}

          {recordedAudio && (
            <div className="audio-playback-section">
              <h3>🎵 Recorded Audio</h3>
              <audio controls src={recordedAudio} className="audio-player"></audio>
              <div className="audio-controls">
                <button
                  type="button"
                  className="send-audio-button"
                  onClick={handleSendAudio}
                  disabled={isTranscribing}
                >
                  {isTranscribing ? (
                    <>
                      <span className="loading-spinner"></span>
                      Transcribing...
                    </>
                  ) : (
                    '📤 Send for Transcription'
                  )}
                </button>
                <button
                  type="button"
                  className="delete-audio-button"
                  onClick={handleDeleteAudio}
                  disabled={isTranscribing}
                >
                  🗑️ Delete
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </form>

        {result && (
          <div className="result-section">
            {/* Overall Result Header */}
            <div className={`overall-result ${result.label === 'metaphor' ? 'result-metaphor' : result.label === 'neutral' ? 'result-neutral' : 'result-normal'}`}>
              <h2>🧾 {isParagraph ? 'Paragraph' : 'Sentence'} Result: {result.label === 'metaphor' ? '🔵 Metaphor' : result.label === 'neutral' ? '⚪ Neutral' : '🟢 Normal'}</h2>
              <div className="confidence-badge">
                Confidence: {(result.confidence * 100).toFixed(2)}%
              </div>
            </div>

            {/* Sentence-level Analysis */}
            {result.sentences && result.sentences.length > 0 && (
              <div className="sentences-container">
                {result.sentences.map((sentence, index) => (
                  <div key={index} className="sentence-card">
                    {/* Only show sentence header for paragraphs */}
                    {isParagraph && (
                      <div className="sentence-header">
                        <h3>Sentence {index + 1}: {sentence.sentence}</h3>
                        <div className="sentence-meta">
                          <span className={`label-badge ${sentence.label === 'metaphor' ? 'badge-metaphor' : 'badge-normal'}`}>
                            {sentence.label === 'metaphor' ? '🔵 Metaphor' : '🟢 Normal'}
                          </span>
                          <span className="confidence-text">
                            Confidence: {(sentence.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Interpretations */}
                    {sentence.interpretations && (
                      <div className="interpretations">
                        <h4>Interpretations:</h4>

                        <div className="interpretation-item">
                          <div className="interpretation-icon">🌐</div>
                          <div className="interpretation-content">
                            <strong>Translation:</strong>
                            <p>{sentence.interpretations.translation}</p>
                          </div>
                        </div>

                        <div className="interpretation-item">
                          <div className="interpretation-icon">💬</div>
                          <div className="interpretation-content">
                            <strong>Literal:</strong>
                            <p>{sentence.interpretations.literal}</p>
                          </div>
                        </div>

                        <div className="interpretation-item">
                          <div className="interpretation-icon">❤️</div>
                          <div className="interpretation-content">
                            <strong>Emotional:</strong>
                            <p>{sentence.interpretations.emotional}</p>
                          </div>
                        </div>

                        <div className="interpretation-item">
                          <div className="interpretation-icon">🧘</div>
                          <div className="interpretation-content">
                            <strong>Philosophical:</strong>
                            <p>{sentence.interpretations.philosophical}</p>
                          </div>
                        </div>

                        <div className="interpretation-item">
                          <div className="interpretation-icon">🌏</div>
                          <div className="interpretation-content">
                            <strong>Cultural:</strong>
                            <p>{sentence.interpretations.cultural}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <footer className="app-footer">
          <p>Supports Hindi (हिंदी), Tamil (தமிழ்), Telugu (తెలుగు), and Kannada (ಕನ್ನಡ)</p>
        </footer>
      </div>

      {/* Virtual Keyboard */}
      {showKeyboard && (
        <VirtualKeyboard
          language={keyboardLang}
          onKeyPress={handleVirtualKeyPress}
          onClose={handleCloseKeyboard}
        />
      )}

      {/* History Modal */}
      {showHistory && (
        <History
          apiBaseUrl={API_BASE_URL}
          onClose={() => setShowHistory(false)}
        />
      )}
    </div>
  );
}

export default App;
