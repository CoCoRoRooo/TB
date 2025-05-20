document.addEventListener('DOMContentLoaded', function () {
    const chatMessages = document.getElementById('chat-messages')
    const userInput = document.getElementById('user-input')
    const sendButton = document.getElementById('send-button')
    const processingTime = document.getElementById('processing-time')
    const queriesList = document.getElementById('queries-list')
    const toggleQueries = document.getElementById('toggle-queries')
    const queriesContainer = document.getElementById('queries-container')
    const documentsContainer = document.getElementById('documents-container')
    const documentsList = document.getElementById('documents-list')
    const toggleDocuments = document.getElementById('toggle-documents')
    const sidebarToggle = document.getElementById('sidebar-toggle')
    const appContainer = document.querySelector('.app-container')
    const sidebar = document.querySelector('.sidebar')

    // Vérifier si on est en mode mobile au chargement
    function checkMobileView() {
        if (window.innerWidth <= 768) {
            appContainer.classList.add('sidebar-collapsed');

            // Mettre à jour l'icône
            const icon = sidebarToggle.querySelector('i');
            icon.classList.remove('fa-chevron-left');
            icon.classList.add('fa-chevron-right');

            // S'assurer que la sidebar est bien masquée
            sidebar.style.transform = 'translateX(-100%)';
        } else {
            // En mode desktop, s'assurer que la sidebar est visible
            sidebar.style.transform = 'translateX(0)';
        }
    }

    // Exécuter au chargement
    checkMobileView();

    // Exécuter aussi lors du redimensionnement de la fenêtre
    window.addEventListener('resize', checkMobileView);

    window.addEventListener('resize', function () {
        const chatInputContainer = document.querySelector('.chat-input-container');

        if (window.innerWidth <= 768 || appContainer.classList.contains('sidebar-collapsed')) {
            chatInputContainer.style.left = '0';
        } else {
            chatInputContainer.style.left = '280px';
        }
    });

    // Toggle sidebar avec correction pour mobile
    sidebarToggle.addEventListener('click', function () {
        appContainer.classList.toggle('sidebar-collapsed');
        const icon = sidebarToggle.querySelector('i');

        if (appContainer.classList.contains('sidebar-collapsed')) {
            icon.classList.remove('fa-chevron-left');
            icon.classList.add('fa-chevron-right');
            // Animation fluide pour la sidebar
            sidebar.style.transform = 'translateX(-100%)';
        } else {
            icon.classList.remove('fa-chevron-right');
            icon.classList.add('fa-chevron-left');
            // Animation fluide pour la sidebar
            sidebar.style.transform = 'translateX(0)';
        }

        const chatInputContainer = document.querySelector('.chat-input-container');
        if (appContainer.classList.contains('sidebar-collapsed')) {
            // Si la sidebar est fermée
            chatInputContainer.style.left = '0';
        } else {
            // Si la sidebar est ouverte
            if (window.innerWidth > 768) {
                chatInputContainer.style.left = '280px';
            } else {
                chatInputContainer.style.left = '0';
            }
        }
    });

    // Fonction pour formater le texte markdown
    function formatMarkdown(text) {
        // Formatage spécial pour les sections du format de réponse attendu
        text = text.replace(/🔍\s*\*\*Analyse du problème\*\*\s*:/g, '<div class="problem-analysis"><strong>🔍 Analyse du problème :</strong>')
        text = text.replace(/✅\s*\*\*Vérifications préalables recommandées\*\*\s*:/g, '</div><div class="checks"><strong>✅ Vérifications préalables recommandées :</strong>')
        text = text.replace(/📝\s*\*\*Procédure détaillée proposée\*\*\s*:/g, '</div><div class="procedure"><strong>📝 Procédure détaillée proposée :</strong>')
        text = text.replace(/💡\s*\*\*Conseils supplémentaires ou précautions à prendre\*\*\s*:/g, '</div><div class="tips"><strong>💡 Conseils supplémentaires ou précautions à prendre :</strong>')
        text = text.replace(/🔗\s*\*\*Sources consultées\*\*\s*:/g, '</div><div class="sources"><strong>🔗 Sources consultées :</strong>')

        // Ajouter la div de fermeture si nécessaire
        if (text.includes('<div class="sources">')) {
            text += '</div>'
        }

        // Utiliser marked pour le reste du formatage markdown
        let formattedText = marked.parse(text)

        // Sanitiser le HTML pour éviter les injections XSS
        return DOMPurify.sanitize(formattedText)
    }

    // Fonction pour ajouter un message au chat avec avatar
    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div')
        messageDiv.className = `message ${isUser ? 'message-user' : 'message-bot'}`

        // Créer l'avatar
        const avatarDiv = document.createElement('div')
        avatarDiv.className = 'message-avatar'

        // Icône différente selon l'utilisateur ou le bot
        const iconElement = document.createElement('i')
        iconElement.className = isUser ? 'fas fa-user' : 'fas fa-robot'
        avatarDiv.appendChild(iconElement)

        // Créer le contenu du message
        const contentDiv = document.createElement('div')
        contentDiv.className = 'message-content'

        const innerDiv = document.createElement('div')
        innerDiv.className = 'markdown-body'

        if (isUser) {
            innerDiv.textContent = content
        } else {
            // Pour les messages du bot, nous utiliserons l'effet de typing
            addTypingEffect(innerDiv, content)
        }

        contentDiv.appendChild(innerDiv)

        // Assembler le message
        messageDiv.appendChild(avatarDiv)
        messageDiv.appendChild(contentDiv)

        chatMessages.appendChild(messageDiv)
        chatMessages.scrollTop = chatMessages.scrollHeight

        // Animation d'entrée
        setTimeout(() => {
            messageDiv.style.opacity = '1'
            messageDiv.style.transform = 'translateY(0)'
        }, 10)

        return { messageDiv, innerDiv }
    }

    // Fonction pour ajouter l'effet de frappe en temps réel
    async function addTypingEffect(element, content) {
        // Paramètres ajustés pour un effet visuel plus naturel 
        const minDelay = 10;   // Délai minimum entre les caractères (ms)
        const maxDelay = 25;   // Délai maximum entre les caractères (ms)
        const chunkSize = 3;   // Nombre de caractères à traiter par itération
        const fastModeThreshold = 800; // Seuil à partir duquel on accélère le traitement

        // Mode accéléré pour les contenus longs
        const fastMode = content.length > fastModeThreshold;

        // Traitement du contenu
        let currentText = '';

        // Version simplifiée pour cette démo
        if (fastMode) {
            // Pour les longs textes, on traite par chunks plus grands
            for (let i = 0; i < content.length; i += chunkSize * 2) {
                const endPos = Math.min(i + chunkSize * 2, content.length);
                currentText += content.substring(i, endPos);
                element.innerHTML = formatMarkdown(currentText);
                chatMessages.scrollTop = chatMessages.scrollHeight;

                await new Promise(resolve => setTimeout(resolve, 5));
            }
        } else {
            // Mode normal avec effet de frappe visible
            for (let i = 0; i < content.length; i += chunkSize) {
                const endPos = Math.min(i + chunkSize, content.length);
                currentText += content.substring(i, endPos);
                element.innerHTML = formatMarkdown(currentText);
                chatMessages.scrollTop = chatMessages.scrollHeight;

                await new Promise(resolve => setTimeout(resolve,
                    Math.floor(Math.random() * (maxDelay - minDelay + 1)) + minDelay));
            }
        }

        // Assurer que le formatage final est appliqué
        element.innerHTML = formatMarkdown(content);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Fonction pour ajouter un message de chargement
    function addLoadingMessage() {
        const loadingDiv = document.createElement('div')
        loadingDiv.className = 'message message-bot'
        loadingDiv.id = 'loading-message'

        // Créer l'avatar
        const avatarDiv = document.createElement('div')
        avatarDiv.className = 'message-avatar'

        const iconElement = document.createElement('i')
        iconElement.className = 'fas fa-robot'
        avatarDiv.appendChild(iconElement)

        // Créer le contenu du message
        const contentDiv = document.createElement('div')
        contentDiv.className = 'message-content'

        const innerDiv = document.createElement('div')
        innerDiv.className = 'flex items-center'
        innerDiv.innerHTML = `
              <div class="flex items-center space-x-2">
                  <svg class="animate-spin h-4 w-4 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span class="text-indigo-700 font-medium">Analyse en cours</span>
                  <span class="loading-dots text-indigo-700"></span>
              </div>
          `

        contentDiv.appendChild(innerDiv)

        // Assembler le message
        loadingDiv.appendChild(avatarDiv)
        loadingDiv.appendChild(contentDiv)

        chatMessages.appendChild(loadingDiv)
        chatMessages.scrollTop = chatMessages.scrollHeight
    }

    // Fonction pour supprimer le message de chargement
    function removeLoadingMessage() {
        const loadingMessage = document.getElementById('loading-message')
        if (loadingMessage) {
            loadingMessage.remove()
        }
    }

    // Fonction pour afficher les requêtes générées
    function displayQueries(queries) {
        queriesList.innerHTML = ''

        // Ajouter les requêtes avec une légère animation
        queries.forEach((query, index) => {
            setTimeout(() => {
                const li = document.createElement('li')
                li.textContent = query
                li.style.opacity = '0'
                li.style.transform = 'translateY(5px)'
                queriesList.appendChild(li)

                // Déclencher l'animation
                setTimeout(() => {
                    li.style.transition = 'all 0.3s ease'
                    li.style.opacity = '1'
                    li.style.transform = 'translateY(0)'
                }, 50)
            }, index * 100) // Ajouter un délai pour chaque élément
        })
    }

    // Gestion du toggle pour les requêtes avec animation
    toggleQueries.addEventListener('click', function () {
        if (queriesContainer.style.display === 'none' || queriesContainer.style.display === '' || queriesContainer.style.opacity === '0') {
            queriesContainer.style.display = 'block';
            setTimeout(() => {
                queriesContainer.style.opacity = '1';
                queriesContainer.style.transform = 'translateY(0)';
            }, 50);
        } else {
            queriesContainer.style.opacity = '0';
            queriesContainer.style.transform = 'translateY(-10px)';
            setTimeout(() => {
                queriesContainer.style.display = 'none';
            }, 300);
        }
    });

    // Fonction pour afficher les documents récupérés
    function displayDocuments(documents) {
        documentsList.innerHTML = ''

        if (!documents || documents.length === 0) {
            const emptyMsg = document.createElement('div')
            emptyMsg.className = 'text-gray-500 italic text-center py-4'
            emptyMsg.textContent = 'Aucun document récupéré'
            documentsList.appendChild(emptyMsg)
            return
        }

        const docsArray = documents.split('---').filter(doc => doc.trim() !== '')

        // Ajouter les documents avec une légère animation
        docsArray.forEach((doc, index) => {
            setTimeout(() => {
                const docElement = document.createElement('div')
                docElement.className = 'document-item bg-white rounded-lg p-3 border border-blue-100 shadow-sm'
                docElement.style.opacity = '0'
                docElement.style.transform = 'translateY(5px)'

                // Formater le contenu du document
                const formattedContent = formatDocumentContent(doc)
                docElement.innerHTML = formattedContent

                documentsList.appendChild(docElement)

                // Déclencher l'animation
                setTimeout(() => {
                    docElement.style.transition = 'all 0.3s ease'
                    docElement.style.opacity = '1'
                    docElement.style.transform = 'translateY(0)'
                }, 50)
            }, index * 150) // Ajouter un délai pour chaque élément
        })
    }

    // Formatage du contenu du document
    function formatDocumentContent(docText) {
        let content = docText.trim()

        // Extraire le contenu et les métadonnées
        const contentMatch = content.match(/📄 Contenu :\n([\s\S]*?)(?:\n\n🔖 Métadonnées :|$)/)
        const metadataMatch = content.match(/🔖 Métadonnées :\n([\s\S]*)/)

        let docContent = contentMatch ? contentMatch[1].trim() : ''
        let metadata = metadataMatch ? metadataMatch[1].trim() : ''

        // Extraire et mettre en forme l'URL si présente
        let url = ''
        if (metadata) {
            const urlMatch = metadata.match(/url: ([^\n]+)/)
            if (urlMatch && urlMatch[1] && urlMatch[1] !== 'undefined') {
                url = urlMatch[1].trim()
            }

            // Extraire le titre si présent
            const titleMatch = metadata.match(/titre: ([^\n]+)/) || metadata.match(/title: ([^\n]+)/)
            let title = titleMatch && titleMatch[1] !== 'undefined' ? titleMatch[1].trim() : 'Document sans titre'

            // HTML pour le document formaté
            return `
                  <div class="document-header flex items-center justify-between mb-2">
                    <h3 class="font-medium text-blue-800">${title}</h3>
                    <span class="text-xs text-gray-500">${url ? `<a href="${url}" target="_blank" class="text-blue-600 hover:underline flex items-center" title="${url}"><i class="fas fa-external-link-alt mr-1"></i>Source</a>` : ''}</span>
                  </div>
                  <div class="document-content text-xs text-gray-700 border-t border-blue-100 pt-2">
                      ${docContent.length > 150 ? docContent.substring(0, 150) + '...' : docContent}
                  </div>
                  <button class="view-more-btn text-xs text-blue-600 mt-2 hover:underline flex items-center">
                      <i class="fas fa-eye mr-1"></i> Voir détails
                  </button>
                  <div class="document-full-content hidden bg-blue-50 mt-2 p-2 rounded border border-blue-100 text-xs">
                      <div class="mb-2"><strong>Contenu complet:</strong></div>
                      ${docContent.replace(/\n/g, '<br>')}
                      ${metadata ? `<div class="mt-2 pt-2 border-t border-blue-200"><strong>Métadonnées:</strong><br>${metadata.replace(/\n/g, '<br>')}</div>` : ''}
                  </div>
              `
        } else {
            return `<div class="text-gray-700">${docContent}</div>`
        }
    }

    // Gestion du toggle pour les documents
    toggleDocuments.addEventListener('click', function () {
        if (documentsContainer.style.display === 'none' || documentsContainer.style.display === '' || documentsContainer.style.opacity === '0') {
            documentsContainer.style.display = 'block';
            setTimeout(() => {
                documentsContainer.style.opacity = '1';
                documentsContainer.style.transform = 'translateY(0)';
                toggleDocuments.innerHTML = '<i class="fas fa-folder-open"></i>';
            }, 50);
        } else {
            documentsContainer.style.opacity = '0';
            documentsContainer.style.transform = 'translateY(-10px)';
            setTimeout(() => {
                documentsContainer.style.display = 'none';
                toggleDocuments.innerHTML = '<i class="fas fa-folder-closed"></i>';
            }, 300);
        }
    });

    // Délégation d'événements pour les boutons "Voir détails"
    documentsList.addEventListener('click', function (e) {
        if (e.target.classList.contains('view-more-btn') || e.target.parentElement.classList.contains('view-more-btn')) {
            const button = e.target.classList.contains('view-more-btn') ? e.target : e.target.parentElement
            const fullContent = button.nextElementSibling

            if (fullContent.classList.contains('hidden')) {
                fullContent.classList.remove('hidden')
                button.innerHTML = '<i class="fas fa-eye-slash mr-1"></i> Masquer détails'
            } else {
                fullContent.classList.add('hidden')
                button.innerHTML = '<i class="fas fa-eye mr-1"></i> Voir détails'
            }
        }
    })

    // Fonction pour envoyer un message avec feedback visuel amélioré
    async function sendMessage() {
        const message = userInput.value.trim()
        if (!message) return

        // Désactiver l'input pendant l'envoi
        userInput.disabled = true
        sendButton.disabled = true
        sendButton.classList.add('opacity-50')

        // Vider l'affichage des documents à chaque nouvelle question
        documentsList.innerHTML = ''
        documentsContainer.style.display = 'none'
        toggleDocuments.innerHTML = '<i class="fas fa-folder-closed"></i>'

        // Afficher le message de l'utilisateur
        addMessage(message, true)
        userInput.value = ''

        // Afficher le message de chargement
        addLoadingMessage()

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            })

            const data = await response.json()

            // Supprimer le message de chargement
            removeLoadingMessage()

            if (data.error) {
                addMessage(`Erreur: ${data.error}`)
            } else {
                // Afficher la réponse avec l'effet de typing
                addMessage(data.response)

                // Afficher les requêtes générées
                if (data.queries && data.queries.length > 0) {
                    displayQueries(data.queries)
                }

                // Afficher les documents récupérés
                if (data.documents) {
                    displayDocuments(data.documents)
                }

                // Afficher le temps de traitement avec animation
                const oldTime = processingTime.textContent
                processingTime.textContent = data.processing_time || '-'

                // Effet visuel pour le changement de temps
                processingTime.classList.add('text-green-600')
                processingTime.classList.add('font-bold')

                setTimeout(() => {
                    processingTime.classList.remove('text-green-600')
                    processingTime.classList.remove('font-bold')
                }, 1000)
            }
        } catch (error) {
            removeLoadingMessage()
            addMessage(`Erreur de connexion: ${error.message}`)
        } finally {
            // Réactiver l'input
            userInput.disabled = false
            sendButton.disabled = false
            sendButton.classList.remove('opacity-50')
            userInput.focus()
        }
    }

    // Événements pour envoyer un message
    sendButton.addEventListener('click', sendMessage)
    userInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage()
        }
    })

    // Animation de la zone d'input lors du focus
    userInput.addEventListener('focus', function () {
        this.parentElement.classList.add('ring-4')
    })

    userInput.addEventListener('blur', function () {
        this.parentElement.classList.remove('ring-4')
    })

    // Mettre le focus sur l'input au chargement
    userInput.focus()

    // Initialiser l'interface - masquer les requêtes et documents au démarrage
    queriesContainer.style.display = 'none'
    toggleQueries.innerHTML = '<i class="fas fa-search"></i>'

    documentsContainer.style.display = 'none'
    toggleDocuments.innerHTML = '<i class="fas fa-folder-closed"></i>'
});