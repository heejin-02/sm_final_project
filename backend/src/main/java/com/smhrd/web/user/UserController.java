package com.smhrd.web.user;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/user")
public class UserController {
	
	@Autowired
    private UserService userService;

    // 로그인 유저의 농장 목록 조회
    @GetMapping("/farms/{userPhone}")
    public ResponseEntity<List<UserDTO>> getFarmsByUser(@PathVariable String userPhone) {
        List<UserDTO> farms = userService.getUserFarms(userPhone);
        return ResponseEntity.ok(farms);
    }
}
