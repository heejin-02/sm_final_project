package com.smhrd.web.admin.userInsert;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface UserInsertMapper {
    // 이미 회원 존재하는지 확인
    int checkUserExists(@Param("userPhone") String userPhone);

    // 회원 등록
    int insertUser(UserInsertDTO user);
}
