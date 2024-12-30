import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [filename, setFilename] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleFilenameChange = (e) => {
    setFilename(e.target.value);
  };

  const handleQuestionChange = (e) => {
    setQuestion(e.target.value);
  };

  const uploadFile = async () => {
    if (!file) {
      setError("Please select a PDF file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:8000/upload/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      alert(response.data.message);
      setFilename(response.data.filename); // Save filename after upload
    } catch (err) {
      setError(err.response?.data?.detail || "Error uploading file");
    }
  };

  const askQuestion = async () => {
    if (!filename || !question) {
      setError("Please upload a file and enter a question.");
      return;
    }

    try {
      const response = await axios.post(
        "http://localhost:8000/question/",
        new URLSearchParams({
          filename: filename,
          question: question,
        }),
        {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
        }
      );
      setAnswer(response.data.answer);
    } catch (err) {
      setError(err.response?.data?.detail || "Error processing question");
    }
  };

  return (
    <div className="App">
      <h1>PDF Q&A Application</h1>

      <motion.div
        className="upload-section"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h3>Upload a PDF</h3>
        <input type="file" accept=".pdf" onChange={handleFileChange} />
        <button onClick={uploadFile}>Upload PDF</button>
      </motion.div>

      <motion.div
        className="question-section"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
      >
        <h3>Ask a Question</h3>
        <input
          type="text"
          placeholder="Enter your question"
          value={question}
          onChange={handleQuestionChange}
        />
        <button onClick={askQuestion}>Ask</button>
      </motion.div>

      {error && <p className="error">{error}</p>}

      {answer && (
        <motion.div
          className="answer-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <h3>Answer</h3>
          <p>{answer}</p>
        </motion.div>
      )}
    </div>
  );
}

export default App;