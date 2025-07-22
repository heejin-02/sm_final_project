package com.smhrd.web.qcClassification;

import org.springframework.stereotype.Service;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class QcClassificationService {
    
    private final QcClassificationMapper mapper;

    public void saveClassification(QcClassificationDTO dto) {
        mapper.insertClassification(dto);
    }
}

