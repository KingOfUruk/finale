document.addEventListener('DOMContentLoaded', function() {
    console.log("Document loaded, initializing script...");
    
    const loginForm = document.getElementById('loginForm');
    const togglePassword = document.getElementById('togglePassword');
    const passwordInput = document.getElementById('loginPassword');
    const notification = document.getElementById('notification');
    const notificationMessage = document.getElementById('notificationMessage');
    const closeNotification = document.querySelector('.close-notification');

    if (!loginForm) {
        console.error("Login form not found!");
        return;
    }

    if (!togglePassword) {
        console.error("Toggle password button not found!");
    }

    // Toggle password visibility
    if (togglePassword && passwordInput) {
        togglePassword.addEventListener('click', function() {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            
            const eyeIcon = this.querySelector('i');
            if (eyeIcon) {
                eyeIcon.classList.toggle('fa-eye');
                eyeIcon.classList.toggle('fa-eye-slash');
            }
        });
    }

    // Close notification
    if (closeNotification && notification) {
        closeNotification.addEventListener('click', function() {
            notification.classList.remove('show');
        });
    }

    // Show notification
    function showNotification(message, isSuccess = true) {
        if (!notification || !notificationMessage) {
            console.error("Notification elements not found!");
            alert(message);
            return;
        }
        
        notification.className = isSuccess ? 'notification success' : 'notification error';
        notificationMessage.textContent = message;
        notification.classList.add('show');
        
        setTimeout(() => {
            notification.classList.remove('show');
        }, 5000);
    }

    // Handle login form submission
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        console.log("Login form submitted");
        
        const username = document.getElementById('loginUsername').value;
        const password = document.getElementById('loginPassword').value;
        const rememberMe = document.getElementById('rememberMe').checked;
        
        if (!username || !password) {
            showNotification('Veuillez saisir le nom d\'utilisateur et le mot de passe', false);
            return;
        }
        
        const submitBtn = loginForm.querySelector('button');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Authentification...';
        submitBtn.disabled = true;
        
        try {
            console.log("Sending login request to server...");
            const response = await fetch('http://localhost:5000/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                credentials: 'include',
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            });
            
            console.log("Login response status:", response.status);
            console.log("Login response headers:", [...response.headers]);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Login response data:", data);
            
            if (data.success) {
                showNotification('Authentification réussie! Redirection...', true);
                
                if (rememberMe) {
                    localStorage.setItem('rememberMe', 'true');
                    localStorage.setItem('username', username);
                } else {
                    localStorage.removeItem('rememberMe');
                    localStorage.removeItem('username');
                }
                
                // Verify session before redirecting
                const authCheck = await fetch('http://localhost:5000/check-auth', {
                    credentials: 'include'
                });
                console.log("Auth check response status:", authCheck.status);
                console.log("Auth check response headers:", [...authCheck.headers]);
                
                if (!authCheck.ok) {
                    throw new Error(`HTTP error! status: ${authCheck.status}`);
                }
                
                const authData = await authCheck.json();
                console.log("Auth check after login:", authData);
                
                if (authData.success) {
                    setTimeout(() => {
                        console.log("Redirecting to /home");
                        window.location.href = '/home';
                    }, 1500);
                } else {
                    throw new Error('Session not properly set after login: ' + (authData.message || 'Unknown error'));
                }
            } else {
                showNotification(data.message || 'Identifiants invalides. Veuillez réessayer.', false);
            }
        } catch (error) {
            console.error('Login error:', error);
            showNotification('Erreur lors de l\'authentification: ' + error.message, false);
        } finally {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    });

    // Check if user is already logged in
    async function checkAuth() {
        try {
            console.log("Checking authentication status...");
            const response = await fetch('http://localhost:5000/check-auth', {
                credentials: 'include'
            });
            
            console.log("Check-auth response status:", response.status);
            console.log("Check-auth response headers:", [...response.headers]);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Check-auth response data:", data);
            
            if (data.success) {
                console.log("User is authenticated, redirecting to /home");
                window.location.href = '/home';
            } else {
                console.log("User not authenticated, staying on login page");
                const rememberMe = localStorage.getItem('rememberMe');
                const username = localStorage.getItem('username');
                
                if (rememberMe === 'true' && username) {
                    document.getElementById('loginUsername').value = username;
                    document.getElementById('rememberMe').checked = true;
                }
            }
        } catch (error) {
            console.error('Erreur de vérification d\'authentification:', error);
            showNotification('Erreur lors de la vérification de l\'authentification: ' + error.message, false);
        }
    }

    // Add subtle hover effect to inputs
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            const icon = input.parentElement.querySelector('i');
            if (icon) {
                icon.style.transform = 'translateY(-50%) scale(1.2)';
            }
        });
        
        input.addEventListener('blur', () => {
            const icon = input.parentElement.querySelector('i');
            if (icon) {
                icon.style.transform = 'translateY(-50%)';
            }
        });
    });

    // Initialize the page
    checkAuth();
    console.log("Script initialization complete");
});
