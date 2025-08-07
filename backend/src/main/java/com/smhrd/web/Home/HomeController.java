package com.smhrd.web.Home;

import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/home")
@Tag(name = "Home Controller", description = "MAIN 화면 API")
public class HomeController {

    @Autowired
    private HomeService homeService;

    @GetMapping("/")
    @Operation(summary = "홈 화면입니다.")
    public Map<String, Object> home() {
        Map<String, Object> response = new HashMap<>();
        response.put("message", "백구백구 109의 home 입니다.");
        return response;
    }

    @PostMapping("/loginCheck")
	@Operation(summary = "로그인 처리 - 성공 시 사용자 정보를 보여줍니다.")
    public ResponseEntity<?> loginCheck(@RequestParam("id") String id,
                                        @RequestParam("pw") String pw,
                                        HttpSession session) {

        HomeDTO result = homeService.login(id, pw);

        if (result != null) {
            session.setAttribute("loginId", result.getUserPhone());
            session.setAttribute("userName", result.getUserName());

            Map<String, Object> res = new HashMap<>();
            res.put("userName", result.getUserName());
            res.put("userPhone", result.getUserPhone());

            if ("admin".equals(id) && "admin".equals(pw)) {
                session.setAttribute("role", "admin");
                res.put("role", "admin");
            } else {
                session.setAttribute("role", "user");
                res.put("role", "user");
            }

            return ResponseEntity.ok(res);
        } else {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "아이디 또는 비밀번호가 올바르지 않습니다."));
        }
    }

    @GetMapping("/userPage")
	@Operation(summary = "사용자 페이지 접근 권한을 체크합니다.")
    public ResponseEntity<?> userPage(HttpSession session) {
        String role = (String) session.getAttribute("role");

        if ("user".equals(role)) {
            return ResponseEntity.ok(Map.of("message", "사용자 페이지 접근 허용"));
        } else {
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(Map.of("error", "접근 권한이 없습니다."));
        }
    }

    @GetMapping("/adminPage")
	@Operation(summary = "관리자 페이지 접근 권한을 체크합니다.")
    public ResponseEntity<?> adminPage(HttpSession session) {
        String role = (String) session.getAttribute("role");

        if ("admin".equals(role)) {
            return ResponseEntity.ok(Map.of("message", "관리자 페이지 접근 허용"));
        } else {
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(Map.of("error", "접근 권한이 없습니다."));
        }
    }
    
    @PostMapping("/logout")
    @Operation(summary = "로그아웃 처리 - 세션을 삭제합니다.")
    public ResponseEntity<?> logout(HttpSession session) {
        session.invalidate();
        return ResponseEntity.ok(Map.of("message", "로그아웃이 완료되었습니다."));
    }
    
    @GetMapping("/check-session")
    @Operation(summary = "세션 상태 확인 - 로그인 상태와 사용자 정보를 반환합니다.")
    public ResponseEntity<Map<String, Object>> checkSession(HttpSession session) {
        Map<String, Object> response = new HashMap<>();
        
        String loginId = (String) session.getAttribute("loginId");
        String userName = (String) session.getAttribute("userName");
        String role = (String) session.getAttribute("role");
        
        if (loginId != null && userName != null) {
            // 세션이 유효한 경우
            Map<String, Object> user = new HashMap<>();
            user.put("userPhone", loginId);
            user.put("userName", userName);
            user.put("role", role);
            
            response.put("isAuthenticated", true);
            response.put("user", user);
            
            System.out.println("유효한 세션 발견: " + userName + " (" + role + ")");
        } else {
            // 세션이 없거나 만료된 경우
            response.put("isAuthenticated", false);
            response.put("user", null);
            
            System.out.println("유효한 세션 없음");
        }
        
        return ResponseEntity.ok(response);
    }    


}
