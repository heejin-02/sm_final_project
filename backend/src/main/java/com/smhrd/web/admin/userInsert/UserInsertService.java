package com.smhrd.web.admin.userInsert;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class UserInsertService {

    @Autowired
    private UserInsertMapper mapper;

    public boolean insertUser(UserInsertDTO dto) {
        // 이미 회원 존재하면 가입 실패
        int exists = mapper.checkUserExists(dto.getUserPhone());
        if (exists > 0) {
            return false;
        }
        // 회원 등록
        int result = mapper.insertUser(dto);
        return result > 0;
    }
}
