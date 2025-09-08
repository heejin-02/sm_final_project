import React from 'react';

export default function AlertModal({ message, onClose }) {
  return (
    <div className='modal-overlay feedback-modal' onClick={onClose}>
      <div className='modal' onClick={(e) => e.stopPropagation()}>
        <div className='modal-header'>
          <button onClick={onClose} className='modal-close-btn'>
            ×
          </button>
        </div>
        <div className='modal-body'>
          <p>{message}</p>
        </div>
        <div className='btn-wrap'>
          <button onClick={onClose} className='btn btn-primary'>
            확인
          </button>
        </div>
      </div>
    </div>
  );
}
