/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

export default function App() {
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);

  const checkMessage = async () => {
    // API call placeholder
    setResult({ label: 'Ham', confidence: 0.95 });
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">SMS Spam Classifier</h1>
      <textarea 
        className="w-full border p-2 mb-2"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Enter SMS message..."
      />
      <button className="bg-blue-500 text-white p-2" onClick={checkMessage}>Check Message</button>
      {result && <div className="mt-4 p-4 border">{result.label} (Confidence: {result.confidence})</div>}
    </div>
  );
}
