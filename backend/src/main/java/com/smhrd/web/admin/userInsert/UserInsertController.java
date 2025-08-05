package com.smhrd.web.admin.userInsert;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/admin/user")
@Tag(name = "User Insert Controller", description = "사용자 추가 API")
public class UserInsertController {

    @Autowired
    private UserInsertService Service;

    @PostMapping("/insert")
    @Operation(summary = "회원을 등록합니다.")
    public ResponseEntity<?> insertUser(@RequestBody UserInsertDTO dto) {
        boolean success = Service.insertUser(dto);
        if (success) {
            return ResponseEntity.ok("회원 등록 성공");
        } else {
            return ResponseEntity.badRequest().body("이미 등록된 회원입니다.");
        }
    }
}
