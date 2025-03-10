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

  // Set initial active model
  document.querySelector(`[data-model="${selectedModelInput.value}"]`).classList.add('bg-gray-100');

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

  if ("SpeechRecognition" in window || "webkitSpeechRecognition" in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      isRecording = true;
      recordBtn.classList.replace("bg-gray-100", "bg-red-100");
      recordBtn.querySelector('i').classList.replace("fa-microphone", "fa-stop");
    };

    recognition.onresult = (event) => {
      let transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join(" ");
      chatInput.value = transcript;
    };

    recognition.onend = () => {
      isRecording ? recognition.start() : stopRecording();
    };

    function startRecording() {
      recognition.start();
    }

    function stopRecording() {
      isRecording = false;
      recognition.stop();
      recordBtn.classList.replace("bg-red-100", "bg-gray-100");
      recordBtn.querySelector('i').classList.replace("fa-stop", "fa-microphone");
    }

    recordBtn.addEventListener("click", () => (isRecording ? stopRecording() : startRecording()));
  } else {
    recordBtn.disabled = true;
  }

  modelItems.forEach((item) => {
    item.addEventListener("click", function () {
      const previousModel = selectedModelInput.value;
      const newModel = this.getAttribute("data-model");

      // Don't do anything if clicking on already selected model
      if (previousModel === newModel) return;

      // Update styling
      document.querySelectorAll(".model-item").forEach((i) => i.classList.remove("bg-gray-100"));
      this.classList.add("bg-gray-100");

      // Update selected model
      selectedModelInput.value = newModel;

      // Show SweetAlert notification
      Swal.fire({
        title: 'Model Changed',
        text: `Switched from ${previousModel} to ${newModel}`,
        icon: 'success',
        toast: true,
        position: 'top-end',
        showConfirmButton: false,
        timer: 3000,
        timerProgressBar: true
      });
    });
  });

  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    appendMessage("user", message);
    chatInput.value = "";

    // fro loading container 
    loadingContainer.style.visibility = "visible";
    loadingContainer.style.opacity = "1";

    fetch("/send_message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message, model: selectedModelInput.value }),
    })
      .then((response) => response.json())
      .then((data) => appendMessage("assistant", data.response, data.audio_url))
      .catch(() => {
        Swal.fire({
          title: 'Error',
          text: 'Failed to process message',
          icon: 'error',
          toast: true,
          position: 'top-end',
          showConfirmButton: false,
          timer: 3000
        });
      })
      .finally(() => {
        loadingContainer.style.visibility = "hidden";
        loadingContainer.style.opacity = "0";
      });
  });


  function appendMessage(sender, text, audio_url = null) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "flex mb-4 " + (sender === "assistant" ? "justify-start" : "justify-end");

    const messageContent = document.createElement("div");
    messageContent.className = sender === "assistant"
      ? "message assistant-message bg-white rounded-lg shadow-sm p-3 max-w-3xl"
      : "message user-message bg-blue-500 text-white rounded-lg shadow-sm p-3 max-w-3xl";

    const messageText = document.createElement("p");
    messageText.className = sender === "assistant" ? "text-gray-800" : "";
    messageText.textContent = text;
    messageContent.appendChild(messageText);

    if (audio_url) {
      const audioContainer = document.createElement("div");
      audioContainer.className = "mt-2";
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = audio_url;
      audio.className = "w-full";
      audio.autoplay = true;
      audioContainer.appendChild(audio);
      messageContent.appendChild(audioContainer);
    }

    messageDiv.appendChild(messageContent);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }


});