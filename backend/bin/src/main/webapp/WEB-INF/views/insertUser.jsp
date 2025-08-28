<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ taglib uri="http://java.sun.com/jsp/jstl/fmt" prefix="fmt" %>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>회원 추가 페이지</title>
</head>
<body>
    <h1>회원 추가</h1>
    <form action="/insertUser" method="post">
        <ul>
            <li>
                <span>휴대폰 번호 (아이디)</span>
                <input type="text" name="userPhone" placeholder="휴대폰 번호를 입력해주세요" class="ipt_tt" maxlength="20" required>
            </li>
            <li>
                <span>비밀번호</span>
                <input type="password" name="userPw" placeholder="비밀번호를 입력해주세요" class="ipt_tt" maxlength="100" required>
            </li>
            <li>
                <span>유저 이름</span>
                <input type="text" name="userName" placeholder="이름을 입력해주세요" class="ipt_tt" maxlength="100" required>
            </li>
            <li>
                <span>가입 날짜</span>
                <input type="date" name="joinedAt" class="ipt_tt" required>
            </li>
            <li>
                <button type="submit">등록하기</button>
            </li>
        </ul>
    </form> 
</body>
</html>
