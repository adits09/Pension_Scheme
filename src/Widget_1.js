import React from 'react';
import './Widget.css';

const Widget_1 = ({ onClick }) => {
  const handleClick = () => {
    // Redirect to specific link for Widget 1
    window.open('https://rajasthan.gov.in/updation-message', '_blank');
    if (onClick) onClick();
  };

  return (
    <div className="widget-container" onClick={handleClick}>
      <div className="widget-icon">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
        </svg>
      </div>
      <h3 className="widget-title">Government Schemes</h3>
      <p className="widget-description">Explore various government schemes and benefits available for citizens</p>
      <div className="widget-arrow">→</div>
    </div>
  );
};

export default Widget_1;
