package com.smhrd.web.Home;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class HomeService {

    @Autowired
    private HomeMapper homeMapper;

    public HomeDTO login(String userPhone, String userPw) {
        HomeDTO loginUser = new HomeDTO();
        loginUser.setUserPhone(userPhone);
        loginUser.setUserPw(userPw);
        return homeMapper.login(loginUser);
    }
}
