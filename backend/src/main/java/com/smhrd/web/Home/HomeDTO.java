package com.smhrd.web.Home;

import lombok.Data;

@Data
public class HomeDTO {
    private String userPhone;
    private String userPw;
    private String userName;
    private String role; // 관리자(admin)인지 아닌지

}
