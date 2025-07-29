package com.smhrd.web.user;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


@Service
public class UserService {

    @Autowired
    private UserMapper userMapper;

    // 유저 휴대폰번호로 농장 리스트 조회
    public List<UserDTO> getFarmsByUserPhone(String userPhone) {
        return userMapper.selectFarmsByUserPhone(userPhone);
    }

    
}
