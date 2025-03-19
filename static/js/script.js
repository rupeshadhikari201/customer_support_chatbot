document.addEventListener("DOMContentLoaded", function () {
  const recordBtn = document.getElementById("record-btn");
  const chatInput = document.getElementById("chat-input");
  const modelItems = document.querySelectorAll(".model-item");
  const selectedModelInput = document.getElementById("selected_model");
  const chatForm = document.getElementById("chat-form");
  const chatBox = document.getElementById("chat-box");
  const loadingContainer = document.getElementById("loading_container");
  const clearChatBtn = document.getElementById("clear-chat-btn");

  let recognition;
  let isRecording = false;
  let recognitionTimeout;

  // Set initial active model
  document.querySelector(`[data-model="${selectedModelInput.value}"]`).classList.add('bg-gray-100');

  // Add evaluation button to the UI
  const inputArea = document.getElementById("input-area");
  const evaluateBtn = document.createElement("button");
  evaluateBtn.id = "evaluate-btn";
  evaluateBtn.className = "bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg mr-2 transition-colors flex items-center";
  evaluateBtn.innerHTML = '<i class="fas fa-chart-bar mr-2"></i>Evaluate';
  inputArea.querySelector(".flex").prepend(evaluateBtn);

  // Add compare models button
  const compareBtn = document.createElement("button");
  compareBtn.id = "compare-btn";
  compareBtn.className = "bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg mr-2 transition-colors flex items-center";
  compareBtn.innerHTML = '<i class="fas fa-balance-scale mr-2"></i>Compare';
  inputArea.querySelector(".flex").prepend(compareBtn);

  // Evaluation button click handler
  evaluateBtn.addEventListener("click", function () {
    Swal.fire({
      title: 'Evaluating RAG System',
      text: 'Please wait while we evaluate the system...',
      allowOutsideClick: false,
      didOpen: () => {
        Swal.showLoading();

        fetch("/evaluate_rag", {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        })
          .then(response => response.json())
          .then(data => {
            if (data.error) {
              Swal.fire({
                title: 'Evaluation Error',
                text: data.error,
                icon: 'error'
              });
              return;
            }

            // Format results for display
            let resultsHtml = '<div class="text-left">';
            for (const [metric, value] of Object.entries(data.results)) {
              resultsHtml += `<p><strong>${metric}:</strong> ${value.toFixed(4)}</p>`;
            }
            resultsHtml += '</div>';

            Swal.fire({
              title: 'Evaluation Results',
              html: resultsHtml,
              icon: 'success'
            });
          })
          .catch(error => {
            console.error("Evaluation error:", error);
            Swal.fire({
              title: 'Evaluation Failed',
              text: 'An error occurred during evaluation',
              icon: 'error'
            });
          });
      }
    });
  });

  // Compare models button click handler
  compareBtn.addEventListener("click", function () {
    Swal.fire({
      title: 'Comparing Models',
      text: 'Please wait while we compare the models...',
      allowOutsideClick: false,
      didOpen: () => {
        Swal.showLoading();

        fetch("/compare_models", {
          method: "GET",
          headers: { "Content-Type": "application/json" }
        })
          .then(response => response.json())
          .then(data => {
            if (data.error) {
              Swal.fire({
                title: 'Comparison Error',
                text: data.error,
                icon: 'error'
              });
              return;
            }

            // Display comparison results with chart
            let resultsHtml = `
            <div class="text-left mb-4">
              <p>Comparison of different models across metrics:</p>
            </div>
            <div class="mb-4">
              <img src="${data.chart_url}" alt="Model Comparison Chart" class="max-w-full h-auto">
            </div>
          `;

            Swal.fire({
              title: 'Model Comparison',
              html: resultsHtml,
              width: 800,
              icon: 'info'
            });
          })
          .catch(error => {
            console.error("Comparison error:", error);
            Swal.fire({
              title: 'Comparison Failed',
              text: 'An error occurred during model comparison',
              icon: 'error'
            });
          });
      }
    });
  });

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