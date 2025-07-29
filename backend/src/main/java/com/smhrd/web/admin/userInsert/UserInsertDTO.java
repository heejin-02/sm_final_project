package com.smhrd.web.admin.userInsert;

import lombok.Data;

@Data
public class UserInsertDTO {
    private String userPhone;
    private String userPw;
    private String userName;
    private String joinedAt;  // 실제 등록시엔 서버에서 sysdate 넣는 게 좋음, 일단 String으로 받음
}
