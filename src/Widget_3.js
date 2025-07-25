import React from 'react';
import './Widget.css';

const Widget_3 = ({ onClick }) => {
  const handleClick = () => {
    // Redirect to specific link for Widget 3
    window.open('https://rajasthan.gov.in/pages/tollfree-contacts?lan=en', '_blank');
    if (onClick) onClick();
  };

  return (
    <div className="widget-container" onClick={handleClick}>
      <div className="widget-icon">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="currentColor">
          <path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/>
        </svg>
      </div>
      <h3 className="widget-title">Contact & Support</h3>
      <p className="widget-description">Get help and support from government officials and departments</p>
      <div className="widget-arrow">→</div>
    </div>
  );
};

export default Widget_3;
