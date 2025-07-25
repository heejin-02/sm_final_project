package com.smhrd.web.admin;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class FarmInsertService {

    @Autowired
    private FarmInsertMapper farmInsertMapper;

    public boolean insertFarm(FarmInsertDTO dto) {
        return farmInsertMapper.insertFarm(dto) > 0;
    }
}
