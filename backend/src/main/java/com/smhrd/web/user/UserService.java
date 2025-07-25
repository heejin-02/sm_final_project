package com.smhrd.web.user;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class UserService {
	
	@Autowired
    private UserMapper userMapper;

    public List<UserDTO> getUserFarms(String userPhone) {
        return userMapper.findFarmsByUserPhone(userPhone);
    }
}
