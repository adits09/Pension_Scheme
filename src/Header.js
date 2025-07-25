import React from 'react';
import './Header.css';
import { useNavigate } from 'react-router-dom';

const Header = () => {
  const navigate = useNavigate();
  
  const handleLogoClick = () => {
    window.open('https://rajasthan.gov.in/', '_blank');
  };

  const handleLoginClick = () => {
    navigate('/login');
  };

  const handleSignupClick = () => {
    navigate('/signup');
  };

  return (
    <header className="header">
      <div className="header-left">
        <div className="logo-container" onClick={handleLogoClick}>
          <img 
            src="raj_logo.png"
            alt="Logo" 
            className="logo" 
          />
        </div>
      </div>
      
      <div className="header-right">
        <button className="auth-button login-button" onClick={handleLoginClick}>
          Log in
        </button>
        <button className="auth-button signup-button" onClick={handleSignupClick}>
          Sign up
        </button>
      </div>
    </header>
  );
};

export default Header;
