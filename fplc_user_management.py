from datetime import datetime
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class UserManagementError(Exception):
    """Base exception for user management operations."""
    pass

class UserAlreadyExistsError(UserManagementError):
    """Exception raised when trying to add a user that already exists."""
    pass

class UserNotFoundError(UserManagementError):
    """Exception raised when a user is not found."""
    pass

class InvalidCredentialsError(UserManagementError):
    """Exception raised for invalid username or password during login."""
    pass

class PermissionDeniedError(UserManagementError):
    """Exception raised when a user does not have the necessary permissions."""
    pass

class UserManager:
    """
    Manages user accounts, including creation, authentication, role management, and permissions.
    """
    def __init__(self):
        self.logger = logging.getLogger('UserManager')
        self.users = {}
        self.current_user = None
        self.user_roles = {  # Renamed user_levels to user_roles for consistency
            'administrator': ['run_method', 'edit_method', 'view_data', 'export_data', 'system_config', 'manage_users'],
            'operator': ['run_method', 'view_data', 'export_data'],
            'viewer': ['view_data']
        }
        self.load_default_users()
        self.logger.info("User Manager initialized with default users.")

    def _hash_password(self, password: str) -> str:
        """Hashes a given password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()

    def load_default_users(self):
        """
        Loads predefined default user accounts into the system.
        """
        self.add_user(username="admin", password="admin123", role="administrator", full_name="System Administrator")
        self.add_user(username="operator", password="op123", role="operator", full_name="System Operator")
        # Note: add_user already logs the addition, so no extra logging here.

    def add_user(self, username: str, password: str, role: str, full_name: str):
        """
        Adds a new user account to the system.

        Args:
            username (str): The unique username for the new account.
            password (str): The plain-text password for the new account.
            role (str): The role of the user (e.g., 'administrator', 'operator', 'viewer').
            full_name (str): The full name of the user.

        Raises:
            UserAlreadyExistsError: If a user with the given username already exists.
            ValueError: If the role is invalid or username/password are empty.
        """
        if not username or not password:
            raise ValueError("Username and password cannot be empty.")
        if username in self.users:
            self.logger.warning(f"Attempt to add existing user: '{username}'")
            raise UserAlreadyExistsError(f"User '{username}' already exists.")
        if role not in self.user_roles:
            raise ValueError(f"Invalid role '{role}'. Valid roles are: {', '.join(self.user_roles.keys())}")

        password_hash = self._hash_password(password)
        self.users[username] = {
            'password_hash': password_hash,
            'role': role,
            'full_name': full_name,
            'created_date': datetime.now().isoformat()
        }
        self.logger.info(f"User '{username}' with role '{role}' added successfully.")

    def verify_user(self, username: str, password: str) -> bool:
        """
        Verifies user credentials against stored hashed passwords.

        Args:
            username (str): The username to verify.
            password (str): The plain-text password to verify.

        Returns:
            bool: True if credentials are valid, False otherwise.
        """
        if username not in self.users:
            self.logger.warning(f"Login attempt for non-existent user: '{username}'")
            return False
        password_hash = self._hash_password(password)
        is_valid = self.users[username]['password_hash'] == password_hash
        if not is_valid:
            self.logger.warning(f"Failed login attempt for user: '{username}' (incorrect password).")
        return is_valid

    def login(self, username: str, password: str) -> bool:
        """
        Logs in a user, setting them as the current active user.

        Args:
            username (str): The username attempting to log in.
            password (str): The password for the user.

        Returns:
            bool: True if login is successful, False otherwise.
        """
        if self.verify_user(username, password):
            self.current_user = username
            self.logger.info(f"User '{username}' (Role: {self.get_user_role(username)}) logged in successfully.")
            return True
        self.logger.warning(f"Login failed for user: '{username}'.")
        return False

    def logout(self):
        """
        Logs out the current active user.
        """
        if self.current_user:
            self.logger.info(f"User '{self.current_user}' logged out.")
            self.current_user = None
        else:
            self.logger.info("No user is currently logged in.")

    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticates user credentials without setting current_user.
        This method is now redundant with verify_user but kept for API compatibility.
        It simply calls verify_user internally.

        Args:
            username (str): The username to authenticate.
            password (str): The plain-text password to authenticate.

        Returns:
            bool: True if credentials are valid, False otherwise.
        """
        self.logger.debug(f"Authenticating user '{username}'.")
        return self.verify_user(username, password)

    def remove_user(self, username: str):
        """
        Removes a user account from the system.

        Args:
            username (str): The username of the account to remove.

        Raises:
            UserNotFoundError: If the user to be removed does not exist.
        """
        if username not in self.users:
            self.logger.warning(f"Attempt to remove non-existent user: '{username}'")
            raise UserNotFoundError(f"User '{username}' not found.")
        
        # Prevent removal of the currently logged-in user to avoid session issues
        if self.current_user == username:
            self.logout() # Log out the user if they are currently logged in

        del self.users[username]
        self.logger.info(f"User '{username}' removed successfully.")

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Changes the password for a specified user.

        Args:
            username (str): The username whose password is to be changed.
            old_password (str): The current plain-text password.
            new_password (str): The new plain-text password.

        Returns:
            bool: True if the password was changed successfully, False otherwise.

        Raises:
            UserNotFoundError: If the user does not exist.
            InvalidCredentialsError: If the old password provided is incorrect.
            ValueError: If the new password is empty.
        """
        if not new_password:
            raise ValueError("New password cannot be empty.")
        if username not in self.users:
            raise UserNotFoundError(f"User '{username}' not found.")

        if self.verify_user(username, old_password):
            self.users[username]['password_hash'] = self._hash_password(new_password)
            self.logger.info(f"Password for user '{username}' changed successfully.")
            return True
        else:
            self.logger.warning(f"Failed to change password for '{username}': Invalid old password.")
            raise InvalidCredentialsError("Invalid old password.")

    def change_user_role(self, username: str, new_role: str):
        """
        Changes the role (access level) for a specified user.

        Args:
            username (str): The username whose role is to be changed.
            new_role (str): The new role (e.g., 'administrator', 'operator', 'viewer').

        Raises:
            UserNotFoundError: If the user does not exist.
            ValueError: If the new role is invalid.
        """
        if username not in self.users:
            raise UserNotFoundError(f"User '{username}' not found.")
        if new_role not in self.user_roles:
            raise ValueError(f"Invalid role '{new_role}'. Valid roles are: {', '.join(self.user_roles.keys())}")

        self.users[username]['role'] = new_role
        self.logger.info(f"User '{username}' role changed to '{new_role}'.")

    def get_user_role(self, username: str) -> str | None:
        """
        Gets the role of a specified user.

        Args:
            username (str): The username to query.

        Returns:
            str | None: The user's role string if found, None otherwise.
        """
        return self.users.get(username, {}).get('role')

    def check_permission(self, required_permission: str) -> bool:
        """
        Checks if the current user has a specific permission.

        Permissions are defined in the `user_roles` dictionary.

        Args:
            required_permission (str): The specific permission to check for (e.g., 'run_method', 'system_config').

        Returns:
            bool: True if the current user has the required permission, False otherwise.

        Raises:
            PermissionDeniedError: If no user is currently logged in.
        """
        
        if not self.current_user:
            self.logger.warning(f"Permission check for '{required_permission}' failed: No user logged in.")
            raise PermissionDeniedError("No user is currently logged in.")
        
        user_role = self.get_user_role(self.current_user)
        if user_role in self.user_roles and required_permission in self.user_roles[user_role]:
            self.logger.debug(f"User '{self.current_user}' has permission '{required_permission}'.")
            return True
        
        self.logger.warning(f"User '{self.current_user}' (Role: {user_role}) does not have permission '{required_permission}'.")
        return False

    def get_all_users_info(self):
        """
        Returns a dictionary containing information about all registered users,
        excluding their password hashes for security.

        Returns:
            dict: A dictionary where keys are usernames and values are
                  dictionaries containing user details (role, full_name, created_date).
        """
        all_users_safe_info = {}
        for username, user_data in self.users.items():
            safe_data = {k: v for k, v in user_data.items() if k != 'password_hash'}
            all_users_safe_info[username] = safe_data
        return all_users_safe_info

# Example Usage:
if __name__ == "__main__":
    user_manager = UserManager()

    # --- Test Login and Permissions ---
    print("\n--- Testing Login and Permissions ---")
    if user_manager.login("admin", "admin123"):
        print("Admin logged in.")
        print(f"Current user role: {user_manager.get_user_role(user_manager.current_user)}")
        print(f"Can admin run method? {user_manager.check_permission('run_method')}")
        print(f"Can admin manage users? {user_manager.check_permission('manage_users')}")
        print(f"Can admin view data? {user_manager.check_permission('view_data')}")
        print(f"Can admin edit method? {user_manager.check_permission('edit_method')}")
        print(f"Can admin non-existent permission? {user_manager.check_permission('non_existent_permission')}")
        user_manager.logout()
    else:
        print("Admin login failed.")

    if user_manager.login("operator", "op123"):
        print("Operator logged in.")
        print(f"Current user role: {user_manager.get_user_role(user_manager.current_user)}")
        print(f"Can operator run method? {user_manager.check_permission('run_method')}")
        print(f"Can operator edit method? {user_manager.check_permission('edit_method')}") # Should be False
        print(f"Can operator view data? {user_manager.check_permission('view_data')}")
        user_manager.logout()
    else:
        print("Operator login failed.")

    print("\n--- Testing Failed Login ---")
    if not user_manager.login("admin", "wrongpassword"):
        print("Failed login for admin with wrong password (as expected).")
    if not user_manager.login("nonexistent", "password"):
        print("Failed login for non-existent user (as expected).")

    # --- Test Adding New User ---
    print("\n--- Testing Adding New User ---")
    try:
        user_manager.add_user("newuser", "newpass", "viewer", "New User Account")
        print("New user 'newuser' added.")
        user_manager.login("newuser", "newpass")
        print(f"New user role: {user_manager.get_user_role('newuser')}")
        print(f"Can newuser view data? {user_manager.check_permission('view_data')}")
        print(f"Can newuser run method? {user_manager.check_permission('run_method')}") # Should be False
        user_manager.logout()
    except UserAlreadyExistsError as e:
        print(e)
    except ValueError as e:
        print(e)

    try:
        user_manager.add_user("newuser", "anotherpass", "viewer", "Duplicate User")
    except UserAlreadyExistsError as e:
        print(f"Caught expected error: {e}")
    
    try:
        user_manager.add_user("testinvalidrole", "password", "invalid_role", "Invalid Role User")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # --- Test Change Password ---
    print("\n--- Testing Change Password ---")
    user_manager.login("operator", "op123")
    try:
        if user_manager.change_password("operator", "op123", "new_op_pass"):
            print("Operator password changed successfully.")
            user_manager.logout()
            if user_manager.login("operator", "new_op_pass"):
                print("Logged in with new operator password.")
                user_manager.logout()
            else:
                print("Failed to login with new operator password.")
        else:
            print("Password change failed unexpectedly.")
    except (UserNotFoundError, InvalidCredentialsError, ValueError) as e:
        print(f"Caught error during password change: {e}")

    try:
        user_manager.change_password("operator", "wrong_old_pass", "should_fail_pass")
    except InvalidCredentialsError as e:
        print(f"Caught expected error for incorrect old password: {e}")
    except UserNotFoundError as e:
        print(f"Caught error: {e}")

    # --- Test Change User Role ---
    print("\n--- Testing Change User Role ---")
    try:
        user_manager.login("admin", "admin123") # Admin needs to be logged in to manage roles (conceptual)
        user_manager.change_user_role("newuser", "operator")
        print("User 'newuser' role changed to 'operator'.")
        user_manager.logout()
        user_manager.login("newuser", "newpass") # Still using old password to login
        print(f"Newuser role after change: {user_manager.get_user_role('newuser')}")
        print(f"Can newuser run method now? {user_manager.check_permission('run_method')}") # Should be True
        user_manager.logout()
    except (UserNotFoundError, ValueError) as e:
        print(f"Caught error during role change: {e}")
    except PermissionDeniedError as e:
        print(f"Caught permission error: {e}") # This might happen if 'manage_users' permission is enforced

    try:
        user_manager.change_user_role("nonexistent", "operator")
    except UserNotFoundError as e:
        print(f"Caught expected error for changing role of non-existent user: {e}")
    
    try:
        user_manager.change_user_role("newuser", "super_admin")
    except ValueError as e:
        print(f"Caught expected error for invalid role: {e}")

    # --- Test Remove User ---
    print("\n--- Testing Remove User ---")
    try:
        user_manager.remove_user("newuser")
        print("User 'newuser' removed.")
    except UserNotFoundError as e:
        print(f"Caught error during user removal: {e}")
    
    try:
        user_manager.login("newuser", "newpass")
    except InvalidCredentialsError:
        print("Login for 'newuser' failed after removal (as expected).")
    except Exception:
        print("Login for 'newuser' failed after removal (as expected).") # Catching generic for user not found

    try:
        user_manager.remove_user("nonexistent_user")
    except UserNotFoundError as e:
        print(f"Caught expected error: {e}")

    print("\n--- All Users Info ---")
    print(user_manager.get_all_users_info())

    print("\n--- Testing PermissionDeniedError for no user logged in ---")
    user_manager.logout() # Ensure no user is logged in
    try:
        user_manager.check_permission('view_data')
    except PermissionDeniedError as e:
        print(f"Caught expected error: {e}")