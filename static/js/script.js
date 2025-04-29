document.addEventListener("DOMContentLoaded", function () {
  const recordBtn = document.getElementById("record-btn");
  const chatInput = document.getElementById("chat-input");
  const modelItems = document.querySelectorAll(".model-item");
  const selectedModelInput = document.getElementById("selected_model");
  const chatForm = document.getElementById("chat-form");
  const chatBox = document.getElementById("chat-box");
  const loadingContainer = document.getElementById("loading_container");
  const clearChatBtn = document.getElementById("clear-chat-btn");

  // Tab elements
  const chatTab = document.getElementById("chat-tab");
  const resultsTab = document.getElementById("results-tab");
  const chatPanel = document.getElementById("chat-panel");
  const resultsPanel = document.getElementById("results-panel");

  // Evaluation elements
  const evaluateBtn = document.getElementById("evaluate-btn");
  const compareModelsBtn = document.getElementById("compare-models-btn");
  const noResultsDiv = document.getElementById("no-results");
  const resultsContainerDiv = document.getElementById("results-container");
  const comparisonContainerDiv = document.getElementById("comparison-container");
  const evaluationLoadingDiv = document.getElementById("evaluation-loading");

  // Charts
  let metricsChart = null;
  let comparisonChart = null;

  let recognition;
  let isRecording = false;
  let recognitionTimeout;

  // Set initial active model
  document.querySelector(`[data-model="${selectedModelInput.value}"]`).classList.add('bg-gray-100');

  // Tab switching functionality
  chatTab.addEventListener("click", function () {
    chatTab.classList.add("text-indigo-600", "border-b-2", "border-indigo-600");
    chatTab.classList.remove("text-gray-500");
    resultsTab.classList.remove("text-indigo-600", "border-b-2", "border-indigo-600");
    resultsTab.classList.add("text-gray-500");

    chatPanel.classList.remove("hidden");
    resultsPanel.classList.add("hidden");
  });

  resultsTab.addEventListener("click", function () {
    resultsTab.classList.add("text-indigo-600", "border-b-2", "border-indigo-600");
    resultsTab.classList.remove("text-gray-500");
    chatTab.classList.remove("text-indigo-600", "border-b-2", "border-indigo-600");
    chatTab.classList.add("text-gray-500");

    resultsPanel.classList.remove("hidden");
    chatPanel.classList.add("hidden");
  });

  // Evaluation button click handler
  evaluateBtn.addEventListener("click", function () {
    // Show evaluation loading indicator
    noResultsDiv.classList.add("hidden");
    resultsContainerDiv.classList.add("hidden");
    comparisonContainerDiv.classList.add("hidden");
    evaluationLoadingDiv.classList.remove("hidden");

    // Switch to results tab
    resultsTab.click();

    fetch("/evaluate_rag", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    })
      .then(response => response.json())
      .then(data => {
        // Hide loading indicator
        evaluationLoadingDiv.classList.add("hidden");

        if (data.error) {
          Swal.fire({
            title: 'Evaluation Error',
            text: data.error,
            icon: 'error'
          });
          noResultsDiv.classList.remove("hidden");
          return;
        }

        // Update results UI
        updateEvaluationResults(data);

        // Show results container
        resultsContainerDiv.classList.remove("hidden");
      })
      .catch(error => {
        console.error("Evaluation error:", error);
        evaluationLoadingDiv.classList.add("hidden");
        noResultsDiv.classList.remove("hidden");

        Swal.fire({
          title: 'Evaluation Failed',
          text: 'An error occurred during evaluation',
          icon: 'error'
        });
      });
  });

  // Compare models button click handler
  compareModelsBtn.addEventListener("click", function () {
    // Show evaluation loading indicator
    noResultsDiv.classList.add("hidden");
    resultsContainerDiv.classList.add("hidden");
    comparisonContainerDiv.classList.add("hidden");
    evaluationLoadingDiv.classList.remove("hidden");

    // Switch to results tab
    resultsTab.click();

    fetch("/compare_models", {
      method: "GET",
      headers: { "Content-Type": "application/json" }
    })
      .then(response => response.json())
      .then(data => {
        // Hide loading indicator
        evaluationLoadingDiv.classList.add("hidden");

        if (data.error) {
          Swal.fire({
            title: 'Comparison Error',
            text: data.error,
            icon: 'error'
          });
          noResultsDiv.classList.remove("hidden");
          return;
        }

        // Update comparison UI
        updateModelComparison(data);

        // Show comparison container
        comparisonContainerDiv.classList.remove("hidden");
      })
      .catch(error => {
        console.error("Comparison error:", error);
        evaluationLoadingDiv.classList.add("hidden");
        noResultsDiv.classList.remove("hidden");

        Swal.fire({
          title: 'Comparison Failed',
          text: 'An error occurred during model comparison',
          icon: 'error'
        });
      });
  });

  // Function to update evaluation results UI
  function updateEvaluationResults(data) {
    // Update metadata
    document.getElementById("result-model").textContent = data.model;
    document.getElementById("result-examples").textContent = data.num_examples;
    document.getElementById("result-method").textContent = data.evaluation_method;
    document.getElementById("result-timestamp").textContent = data.timestamp;

    // Update metrics
    const metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "overall_score"];
    metrics.forEach(metric => {
      const element = document.getElementById(`metric-${metric}`);
      if (element && data.results[metric] !== undefined) {
        element.textContent = (data.results[metric] * 100).toFixed(1) + "%";
      }
    });

    // Create/update the chart
    const ctx = document.getElementById('metrics-chart').getContext('2d');

    // Destroy previous chart if it exists
    if (metricsChart) {
      metricsChart.destroy();
    }

    // Create metrics data for the chart
    const chartMetrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"];
    const chartData = chartMetrics.map(metric => data.results[metric] || 0);

    metricsChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Faithfulness', 'Answer Relevancy', 'Context Recall', 'Context Precision'],
        datasets: [{
          label: 'Metrics',
          data: chartData,
          fill: true,
          backgroundColor: 'rgba(79, 70, 229, 0.2)',
          borderColor: 'rgb(79, 70, 229)',
          pointBackgroundColor: 'rgb(79, 70, 229)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgb(79, 70, 229)'
        }]
      },
      options: {
        scales: {
          r: {
            angleLines: {
              display: true
            },
            suggestedMin: 0,
            suggestedMax: 1
          }
        }
      }
    });
  }

  // Function to update model comparison UI
  function updateModelComparison(data) {
    const ctx = document.getElementById('comparison-chart').getContext('2d');

    // Destroy previous chart if it exists
    if (comparisonChart) {
      comparisonChart.destroy();
    }

    // Extract data for the chart
    const comparisonData = data.comparison;
    const models = Object.keys(comparisonData);
    const metrics = Object.keys(comparisonData[models[0]]).filter(key => key !== 'error');

    // Generate datasets for each metric
    const datasets = metrics.map((metric, index) => {
      // Generate a color based on the index
      const hue = (index * 137) % 360;
      const color = `hsl(${hue}, 70%, 60%)`;

      return {
        label: metric,
        data: models.map(model => comparisonData[model][metric] || 0),
        backgroundColor: `hsla(${hue}, 70%, 60%, 0.7)`,
        borderColor: color,
        borderWidth: 1
      };
    });

    comparisonChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: models,
        datasets: datasets
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 1
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'Model Comparison'
          },
          tooltip: {
            mode: 'index',
            intersect: false
          }
        }
      }
    });

    // If there's a chart URL, show it in an image
    if (data.chart_url) {
      const chartImg = document.createElement('img');
      chartImg.src = data.chart_url;
      chartImg.alt = 'Model Comparison Chart';
      chartImg.className = 'w-full h-auto mt-8';

      // Append the image
      document.getElementById('comparison-container').appendChild(chartImg);
    }
  }

  // Clear chat functionality
  clearChatBtn.addEventListener("click", function () {
    Swal.fire({
      title: 'Clear Chat',
      text: 'Are you sure you want to clear all messages?',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#d33',
      cancelButtonColor: '#3085d6',
      confirmButtonText: 'Yes, clear it!'
    }).then((result) => {
      if (result.isConfirmed) {
        fetch("/clear_chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.status === "success") {
              // Clear the chat box
              chatBox.innerHTML = '';
              // Add the welcome message
              appendMessage("assistant", "How can I help you today?");

              Swal.fire({
                title: 'Cleared!',
                text: 'Chat history has been cleared.',
                icon: 'success',
                toast: true,
                position: 'top-end',
                showConfirmButton: false,
                timer: 3000
              });
            }
          })
          .catch((error) => {
            console.error("Error clearing chat:", error);
            Swal.fire({
              title: 'Error',
              text: 'Failed to clear chat history',
              icon: 'error',
              toast: true,
              position: 'top-end',
              showConfirmButton: false,
              timer: 3000
            });
          });
      }
    });
  });

  // Speech recognition setup
  function setupSpeechRecognition() {
    if ("SpeechRecognition" in window || "webkitSpeechRecognition" in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

      // Create a new instance each time to avoid issues
      function createRecognitionInstance() {
        if (recognition) {
          try {
            recognition.stop();
          } catch (e) {
            console.error("Error stopping previous recognition instance:", e);
          }
        }

        recognition = new SpeechRecognition();

        // Configuration - use shorter segments for better reliability
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = "en-US";
        recognition.maxAlternatives = 1;

        // Set a shorter timeout to avoid network issues
        if (recognitionTimeout) {
          clearTimeout(recognitionTimeout);
        }

        // Event handlers
        recognition.onstart = () => {
          isRecording = true;
          recordBtn.classList.replace("bg-gray-100", "bg-red-100");
          recordBtn.querySelector('i').classList.replace("fa-microphone", "fa-stop");

          // Show recording indicator
          Swal.fire({
            title: 'Listening...',
            text: 'Speak now',
            icon: 'info',
            toast: true,
            position: 'top-end',
            showConfirmButton: false,
            timer: 2000
          });

          // Set a timeout to restart recognition if it gets stuck
          recognitionTimeout = setTimeout(() => {
            if (isRecording) {
              try {
                recognition.stop();
                setTimeout(() => {
                  if (isRecording) {
                    createRecognitionInstance();
                    recognition.start();
                  }
                }, 300);
              } catch (e) {
                console.error("Error in recognition timeout handler:", e);
              }
            }
          }, 8000); // 8 seconds timeout
        };

        recognition.onresult = (event) => {
          let interimTranscript = '';
          let finalTranscript = '';

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcript;
            } else {
              interimTranscript += transcript;
            }
          }

          // Update input field with transcribed text
          if (finalTranscript) {
            chatInput.value = finalTranscript;
            // Reset recognition for next segment
            if (isRecording) {
              try {
                recognition.stop();
                setTimeout(() => {
                  if (isRecording) {
                    createRecognitionInstance();
                    recognition.start();
                  }
                }, 300);
              } catch (e) {
                console.error("Error restarting recognition after final result:", e);
              }
            }
          } else if (interimTranscript) {
            chatInput.value = interimTranscript;
          }
        };

        recognition.onend = () => {
          // Clear the timeout
          if (recognitionTimeout) {
            clearTimeout(recognitionTimeout);
          }

          // If still recording, restart recognition after a short delay
          if (isRecording) {
            setTimeout(() => {
              if (isRecording) {
                try {
                  createRecognitionInstance();
                  recognition.start();
                } catch (e) {
                  console.error("Error restarting recognition:", e);
                  stopRecording();
                }
              }
            }, 300);
          }
        };

        recognition.onerror = (event) => {
          console.error("Speech Recognition Error:", event.error);

          // Only show error message for permanent errors
          if (event.error === 'not-allowed' ||
            event.error === 'service-not-allowed' ||
            event.error === 'audio-capture') {

            let errorMessage = "An error occurred with speech recognition";

            switch (event.error) {
              case 'network':
                errorMessage = "Network error occurred. Trying to reconnect...";
                break;
              case 'not-allowed':
              case 'service-not-allowed':
                errorMessage = "Microphone access denied";
                stopRecording();
                break;
              case 'aborted':
                errorMessage = "Speech recognition aborted";
                break;
              case 'audio-capture':
                errorMessage = "Could not capture audio";
                stopRecording();
                break;
              case 'no-speech':
                errorMessage = "No speech detected";
                break;
              case 'speech-timeout':
                errorMessage = "Speech timeout";
                break;
              case 'language-not-supported':
                errorMessage = "Language not supported";
                break;
              case 'bad-grammar':
                errorMessage = "Bad grammar";
                break;
              default:
                errorMessage = `Error: ${event.error}`;
            }

            // Only show error message for permanent errors
            if (event.error === 'not-allowed' ||
              event.error === 'service-not-allowed' ||
              event.error === 'audio-capture') {
              Swal.fire({
                title: 'Speech Recognition Error',
                text: errorMessage,
                icon: 'error',
                toast: true,
                position: 'top-end',
                showConfirmButton: false,
                timer: 3000
              });
              stopRecording();
            } else if (event.error === 'network') {
              // For network errors, try to restart after a delay
              setTimeout(() => {
                if (isRecording) {
                  try {
                    createRecognitionInstance();
                    recognition.start();
                  } catch (e) {
                    console.error("Error restarting after network error:", e);
                    stopRecording();
                  }
                }
              }, 1000);
            }
          }
        };

        return recognition;
      }

      // Initialize recognition
      recognition = createRecognitionInstance();

    } else {
      console.warn("Speech recognition not supported in this browser");
      // Disable the record button or show a message
      recordBtn.disabled = true;
      recordBtn.title = "Speech recognition not supported in this browser";
      recordBtn.classList.add("opacity-50");

      Swal.fire({
        title: 'Feature Not Available',
        text: 'Speech recognition is not supported in your browser. Please try Chrome, Edge, or Safari.',
        icon: 'warning',
        toast: true,
        position: 'top-end',
        showConfirmButton: false,
        timer: 5000
      });
    }
  }

  function toggleRecording() {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }

  function startRecording() {
    // Request microphone permission first
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(() => {
        try {
          // Create a new instance and start
          setupSpeechRecognition();
          recognition.start();
        } catch (e) {
          console.error("Error starting recognition:", e);
          Swal.fire({
            title: 'Error',
            text: 'Failed to start speech recognition. Please try again.',
            icon: 'error',
            toast: true,
            position: 'top-end',
            showConfirmButton: false,
            timer: 3000
          });
        }
      })
      .catch((error) => {
        console.error("Microphone permission denied:", error);
        Swal.fire({
          title: 'Permission Denied',
          text: 'Microphone access is required for speech recognition',
          icon: 'error',
          toast: true,
          position: 'top-end',
          showConfirmButton: false,
          timer: 3000
        });
      });
  }

  function stopRecording() {
    isRecording = false;

    // Clear any pending timeouts
    if (recognitionTimeout) {
      clearTimeout(recognitionTimeout);
      recognitionTimeout = null;
    }

    try {
      if (recognition) {
        recognition.stop();
      }
    } catch (e) {
      console.error("Error stopping recognition:", e);
    }

    recordBtn.classList.replace("bg-red-100", "bg-gray-100");
    recordBtn.querySelector('i').classList.replace("fa-stop", "fa-microphone");
  }

  // Initialize speech recognition
  setupSpeechRecognition();

  // Record button click handler
  recordBtn.addEventListener("click", toggleRecording);

  // Model selection
  modelItems.forEach((item) => {
    item.addEventListener("click", function () {
      const selectedModel = this.getAttribute("data-model");
      selectedModelInput.value = selectedModel;

      // Update UI to show selected model
      document.querySelectorAll(".model-item").forEach((mi) => {
        mi.classList.remove("bg-gray-100");
      });
      this.classList.add("bg-gray-100");

      // Notify user of model change
      Swal.fire({
        title: 'Model Changed',
        text: `Now using ${selectedModel} model`,
        icon: 'info',
        toast: true,
        position: 'top-end',
        showConfirmButton: false,
        timer: 2000
      });
    });
  });

  // Chat form submission
  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const userMessage = chatInput.value.trim();
    const selectedModel = selectedModelInput.value;

    if (!userMessage) return;

    // Add user message to chat
    appendMessage("user", userMessage);

    // Clear input
    chatInput.value = "";

    // Show loading indicator
    loadingContainer.style.visibility = "visible";

    // Send message to server
    fetch("/send_message", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: userMessage,
        model: selectedModel,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Hide loading indicator
        loadingContainer.style.visibility = "hidden";

        // Add assistant response to chat
        appendMessage("assistant", data.response, data.audio_url);

        // Scroll to bottom
        chatBox.scrollTop = chatBox.scrollHeight;
      })
      .catch((error) => {
        console.error("Error sending message:", error);
        // Hide loading indicator
        loadingContainer.style.visibility = "hidden";

        // Show error message
        Swal.fire({
          title: 'Error',
          text: 'Failed to send message',
          icon: 'error',
          toast: true,
          position: 'top-end',
          showConfirmButton: false,
          timer: 3000
        });
      });
  });

  function appendMessage(sender, text, audio_url = null) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `flex mb-4 ${sender === "assistant" ? "justify-start" : "justify-end"}`;

    const messageContent = document.createElement("div");
    messageContent.className = `message ${sender === "assistant" ? "assistant-message bg-white" : "user-message bg-blue-500 text-white"} rounded-lg shadow-sm p-3 max-w-3xl`;

    const messagePara = document.createElement("p");
    messagePara.className = sender === "assistant" ? "text-gray-800" : "";
    messagePara.textContent = text;

    messageContent.appendChild(messagePara);

    // Add audio player if available
    if (audio_url) {
      const audioDiv = document.createElement("div");
      audioDiv.className = "mt-2";

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = audio_url;
      audio.className = "w-full";

      audioDiv.appendChild(audio);
      messageContent.appendChild(audioDiv);
    }

    messageDiv.appendChild(messageContent);
    chatBox.appendChild(messageDiv);

    // Scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});