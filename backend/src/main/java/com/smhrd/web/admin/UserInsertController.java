package com.smhrd.web.admin;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import io.swagger.v3.oas.annotations.Operation;

@RestController
@RequestMapping("/admin/user")
public class UserInsertController {

    @Autowired
    private UserInsertService Service;

    @Operation(summary = "회원 등록")
    @PostMapping("/insert")
    public ResponseEntity<?> insertUser(@RequestBody UserInsertDTO dto) {
        boolean success = Service.insertUser(dto);
        if (success) {
            return ResponseEntity.ok("회원 등록 성공");
        } else {
            return ResponseEntity.badRequest().body("이미 등록된 회원입니다.");
        }
    }
}
