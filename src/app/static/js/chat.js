
document.addEventListener('DOMContentLoaded', () => {
    const chatWidget = document.getElementById('chat-widget');
    const chatToggle = document.getElementById('chat-toggle');
    const chatIcon = chatToggle.querySelector('.chat-icon');
    const closeIcon = chatToggle.querySelector('.close-icon');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('chat-messages');

    // Toggle Chat
    chatToggle.addEventListener('click', () => {
        chatWidget.classList.toggle('closed');
        chatWidget.classList.toggle('open');
        chatIcon.classList.toggle('hidden');
        closeIcon.classList.toggle('hidden');

        if (chatWidget.classList.contains('open')) {
            chatInput.focus();
        }
    });

    // Send Message
    async function sendMessage(text) {
        if (!text) return;

        // User Message
        appendMessage(text, 'user');
        chatInput.value = '';

        // Loading Indicator
        const loadingId = appendLoading();
        scrollToBottom();

        try {
            const response = await fetch('/api/chat/message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await response.json();

            // Remove Loading
            document.getElementById(loadingId).remove();

            if (data.success) {
                appendMessage(data.response, 'bot');

                // Show related items if any
                if (data.related_items && data.related_items.length > 0) {
                    appendRelatedItems(data.related_items);
                }
            } else {
                appendMessage("Xin lỗi, có lỗi xảy ra. Vui lòng thử lại.", 'bot');
            }
        } catch (e) {
            document.getElementById(loadingId).remove();
            appendMessage("Mất kết nối server.", 'bot');
        }

        scrollToBottom();
    }

    function appendMessage(text, sender) {
        const div = document.createElement('div');
        div.className = `message ${sender} animate-fade-in`;
        div.innerHTML = `
            <div class="message-content">${text}</div>
            <div class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
        `;
        messagesContainer.appendChild(div);
    }

    function appendLoading() {
        const id = 'loading-' + Date.now();
        const div = document.createElement('div');
        div.id = id;
        div.className = 'message bot animate-fade-in';
        div.innerHTML = `<div class="message-content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
        messagesContainer.appendChild(div);
        return id;
    }

    function appendRelatedItems(items) {
        const div = document.createElement('div');
        div.className = 'message bot animate-fade-in';
        const listHtml = items.map(item => `
            <div class="related-item" style="margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;">
                <a href="${item.link}" target="_blank" style="font-weight: 500; font-size: 0.9rem;">${item.title}</a>
                <div style="font-size: 0.75rem; color: #888;">${item.source} - ${item.published}</div>
            </div>
        `).join('');
        div.innerHTML = `<div class="message-content"><strong>Tin liên quan:</strong><br><br>${listHtml}</div>`;
        messagesContainer.appendChild(div);
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Event Listeners
    sendBtn.addEventListener('click', () => sendMessage(chatInput.value.trim()));
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage(chatInput.value.trim());
    });

    // Valid global function for interaction
    window.sendSuggestion = (text) => {
        sendMessage(text);
    };
});
