import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handlePredictClick = () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);

      fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
        
      })
        .then((response) => response.json())
        .then((data) => {
          const fakeProbability = data.prediction_text[0][0];
          const realProbability = data.prediction_text[0][1];

          setPrediction({ fake: fakeProbability, real: realProbability });
        })
        .catch((error) => console.error('Error:', error));
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <div className="container mx-auto p-8 bg-gray-100 rounded-lg shadow-lg text-center">
        <h1 className="text-4xl font-bold underline mb-8 text-blue-600">
          Fake Currency Detection
        </h1>

        <input
          type="file"
          accept=".png, .jpg, .jpeg"
          onChange={handleFileChange}
          ref={fileInputRef}
          className="mb-8 p-4 border border-gray-300 rounded hidden"
          id="fileInput"
        />

        <label
          htmlFor="fileInput"
          className="cursor-pointer bg-blue-500 mx-5 text-white px-6 py-3 rounded-full hover:bg-blue-700"
        >
          Upload Image
        </label>

        <button
          onClick={handlePredictClick}
          className="bg-blue-500 text-white px-6 py-3 rounded-full hover:bg-blue-700 mt-4"
        >
          Predict
        </button>

        {selectedFile && (
  <div className="mt-8 flex flex-col items-center">
    <p className="mb-2 text-lg font-semibold">Selected Image:</p>
    <img
      src={URL.createObjectURL(selectedFile)}
      alt="Selected"
      className="max-w-full max-h-48 rounded"
      style={{ alignSelf: 'center' }}
    />
  </div>
)}

        {prediction && (
          <div className="mt-8 p-4 bg-gray-200 rounded-lg">
            <p className="text-lg font-semibold mb-2 text-blue-600">Probabilities:</p>
            <div className="flex justify-between">
              <div className="flex-1">
                <p className="text-gray-700 font-bold">Fake Probability:</p>
                <p className="text-2xl font-bold text-green-500">
                  {prediction.fake.toFixed(4)}
                </p>
              </div>
              <div className="flex-1">
                <p className="text-gray-700 font-bold ">Real Probability:</p>
                <p className="text-2xl font-bold text-blue-500">
                  {prediction.real.toFixed(4)}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
