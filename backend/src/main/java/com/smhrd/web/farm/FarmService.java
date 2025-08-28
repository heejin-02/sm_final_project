package com.smhrd.web.farm;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


@Service
public class FarmService {

    @Autowired
    private FarmMapper farmMapper;

    public FarmDetailResponseDTO getFarmDetail(Long farmIdx) {
        return farmMapper.selectFarmDetail(farmIdx);
    }
}
